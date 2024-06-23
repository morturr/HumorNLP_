import nltk
import numpy as np
from huggingface_hub import HfFolder
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold, cross_val_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import DatasetDict, Dataset

# import bitsandbytes as bnb
# from peft import (
#     LoraConfig,
#     PeftConfig,
#     get_peft_model,
#     prepare_model_for_kbit_training,
# )

# from datetime import datetime

from data_loader import id2label, label2id, load_dataset, load_cv_dataset
from classify_and_evaluate import evaluate_with_cv
import wandb

import torch

wandb.init(mode='disabled')

DATASET_NAME = 'amazon'
MODEL_ID = "google/flan-t5-base"
REPOSITORY_ID = f"{MODEL_ID.split('/')[1]}-{DATASET_NAME}-text-classification-23-6-test"

training_args = TrainingArguments(
    num_train_epochs=2,
    output_dir=REPOSITORY_ID,
    logging_strategy="steps",
    logging_steps=100,
    # report_to="tensorboard",
    report_to="none",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    # per_device_train_batch_size=1, # All the next 3 rows are for small batch size
    # per_device_eval_batch_size=1,
    # gradient_accumulation_steps=4,
    fp16=True,  # Overflows with fp16
    learning_rate=3e-4,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=REPOSITORY_ID,
    hub_token=HfFolder.get_token(),
    seed=42,
    # bf16=True, # ADDED ON Q-LORA
)

config = AutoConfig.from_pretrained(
    MODEL_ID, num_labels=len(label2id), id2label=id2label, label2id=label2id
)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, config=config) # maybe because of auto?
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def init_model():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_ID, config=config)


def tokenize_function(examples) -> dict:
    """Tokenize the text column in the dataset"""
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred) -> dict:
    """Compute metrics for evaluation"""
    logits, labels = eval_pred
    if isinstance(
        logits, tuple
    ):  # if the model also returns hidden_states or attentions
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def train() -> None:
    """
    Train the model and save it to the Hugging Face Hub.
    """
    print(f'***** Train model: {MODEL_ID} on dataset: {DATASET_NAME} *****')
    dataset = load_dataset("AutoModelForSequenceClassification", DATASET_NAME)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # # Configuring LoRA
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     load_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    #
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     MODEL_ID,
    #     device_map="auto",
    #     trust_remote_code=True,
    #     quantization_config=bnb_config,
    # )
    #
    # model = prepare_model_for_kbit_training(model)
    #
    # lora_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     target_modules=["q", "v"],
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM"
    # )
    #
    # model = get_peft_model(model, lora_config)

    nltk.download("punkt")

    trainer = Trainer(
        # model=model,
        model_init=init_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        compute_metrics=compute_metrics,
    )

    # TRAIN
    trainer.train()

    # SAVE AND EVALUATE
    tokenizer.save_pretrained(REPOSITORY_ID)
    trainer.create_model_card()
    trainer.push_to_hub()
    print(trainer.evaluate())


def train_with_cv() -> None:
    """
    Train the model using cross validation and find the best hyperparameters.
    """
    dataset, kf = load_cv_dataset("AutoModelForSequenceClassification", num_of_split=5, dataset_name=DATASET_NAME)
    for split_idx, split in enumerate(kf.split(dataset['text'], dataset['label'])):
        # Set new repository
        REPOSITORY_ID = f"{MODEL_ID.split('/')[1]}-{DATASET_NAME}-text-classification-split-{split_idx}"
        training_args.output_dir = REPOSITORY_ID
        training_args.hub_model_id = REPOSITORY_ID
        training_args.hub_token = HfFolder.get_token()

        # model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, config=config)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        train = Dataset.from_pandas(dataset.iloc[split[0]])
        test = Dataset.from_pandas(dataset.iloc[split[1]])
        data_dict = DatasetDict()
        data_dict['train'] = train
        data_dict['test'] = test
        tokenized_datasets = data_dict.map(tokenize_function, batched=True)

        nltk.download("punkt")

        trainer = Trainer(
            model_init=init_model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            compute_metrics=compute_metrics,
        )

        # TRAIN
        trainer.train()

        # SAVE AND EVALUATE
        tokenizer.save_pretrained(REPOSITORY_ID)
        trainer.create_model_card()
        trainer.push_to_hub()

        # PREDICT ON 5TH SPLIT
        evaluate_with_cv(data_dict, REPOSITORY_ID, DATASET_NAME)


if __name__ == "__main__":
    # train()
    train_with_cv()