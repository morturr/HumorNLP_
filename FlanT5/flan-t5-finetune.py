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
    # BitsAndBytesConfig,
    HfArgumentParser,
)
from datasets import DatasetDict, Dataset

# import bitsandbytes as bnb
# from peft import (
#     LoraConfig,
#     PeftConfig,
#     get_peft_model,
#     prepare_model_for_kbit_training,
# )

from datetime import datetime

import itertools

from data_loader import id2label, label2id, load_dataset, load_cv_dataset, load_LOO_dataset
from flan_utils import FlanTrainingArguments
from classify_and_evaluate import evaluate_with_cv
import wandb

import torch

wandb.init(mode='disabled')

parser = HfArgumentParser(FlanTrainingArguments)
data_args = parser.parse_args_into_dataclasses()[0]
DATASET_NAME = 'amazon'
MODEL_ID = "google/flan-t5-base"
REPOSITORY_ID = f"{data_args.model_name.split('/')[1]}-{data_args.dataset_name}-text-classification-{datetime.now().date()}"

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
    data_args.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, config=config) # maybe because of auto?
tokenizer = AutoTokenizer.from_pretrained(data_args.model_name)


def init_model():
    return AutoModelForSequenceClassification.from_pretrained(data_args.model_name, config=config)


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
    print(f'***** Train model: {data_args.model_name} on dataset: {data_args.dataset_name} *****')
    dataset = load_dataset(dataset_name=data_args.dataset_name, percent=data_args.samples_percent)
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
    ep = data_args.epochs[0]
    bs = data_args.batch_sizes[0]
    lr = data_args.learning_rates[0]

    for seed in data_args.seeds:
        run_args = {
            'dataset_name': data_args.dataset_name,
            'epoch': ep,
            'batch_size': bs,
            'learning_rate': lr,
            'seed': seed}

        REPOSITORY_ID = f"{data_args.model_name.split('/')[1]}-{data_args.dataset_name}-text-classification-" \
                        f"{datetime.now().date()}-seed-{seed}"

        print(f'training {REPOSITORY_ID}')
        print(f'args = {run_args}')

        training_args_update = {
            "num_train_epochs": ep,
            "per_device_train_batch_size": bs,
            'per_device_eval_batch_size': bs,
            'learning_rate': lr,
            'seed': seed,
            'output_dir': REPOSITORY_ID,
            'hub_model_id': REPOSITORY_ID,
            'hub_token': HfFolder.get_token()
        }

        update_training_arguments(**training_args_update)

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


def train_one_out() -> None:
    """
    Train the model on all datasets in leave-one-out manner
    """
    for test_dataset in data_args.leave_one_out_datasets:
        print(f'***** Train leave-one-out model: {data_args.model_name}. without dataset: {test_dataset} *****')
        dataset = load_LOO_dataset(data_args.leave_one_out_datasets, test_dataset)
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        nltk.download("punkt")
        ep = data_args.epochs[0]
        bs = data_args.batch_sizes[0]
        lr = data_args.learning_rates[0]

        data_dict = DatasetDict()
        data_dict['train'] = dataset['train']
        data_dict['test'] = dataset['test']

        for seed in data_args.seeds:
            run_args = {
                'dataset_name': 'loo_' + test_dataset,
                'epoch': ep,
                'batch_size': bs,
                'learning_rate': lr,
                'seed': seed}

            REPOSITORY_ID = f"{data_args.model_name.split('/')[1]}-loo-{test_dataset}-text-classification-" \
                            f"{datetime.now().date()}-seed-{seed}"

            print(f'training {REPOSITORY_ID}')
            print(f'args = {run_args}')

            training_args_update = {
                "num_train_epochs": ep,
                "per_device_train_batch_size": bs,
                'per_device_eval_batch_size': bs,
                'learning_rate': lr,
                'seed': seed,
                'output_dir': REPOSITORY_ID,
                'hub_model_id': REPOSITORY_ID,
                'hub_token': HfFolder.get_token()
            }

            update_training_arguments(**training_args_update)

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

            evaluate_with_cv(data_dict, REPOSITORY_ID, run_args)


def train_with_cv() -> None:
    """
    Train the model using cross validation and find the best hyperparameters.
    """
    dataset, kf = load_cv_dataset(num_of_split=5, dataset_name=data_args.dataset_name,
                                  percent=data_args.samples_percent)
    for split_idx, split in enumerate(kf.split(dataset['text'], dataset['label'])):
        # Set new repository
        for ep, bs, lr, seed in itertools.product(data_args.epochs, data_args.batch_sizes,
                                                  data_args.learning_rates, data_args.seeds):
            run_args = {
                'dataset_name': data_args.dataset_name,
                'epoch': ep,
                'batch_size': bs,
                'learning_rate': lr,
                'seed': seed}

            REPOSITORY_ID = f"{data_args.model_name.split('/')[1]}-{data_args.dataset_name}-text-classification-" \
                            f"{datetime.now().date()}"

            print(f'training {REPOSITORY_ID}')
            print(f'args = {run_args}')

            training_args_update = {
                "num_train_epochs": ep,
                "per_device_train_batch_size": bs,
                'per_device_eval_batch_size': bs,
                'learning_rate': lr,
                'seed': seed,
                'output_dir': REPOSITORY_ID,
                'hub_model_id': REPOSITORY_ID,
                'hub_token': HfFolder.get_token()
            }

            update_training_arguments(**training_args_update)

            # model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, config=config)
            tokenizer = AutoTokenizer.from_pretrained(data_args.model_name)

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
            evaluate_with_cv(data_dict, REPOSITORY_ID, run_args)


def update_training_arguments(**kwargs):
    global training_args
    for key, value in kwargs.items():
        if hasattr(training_args, key):
            setattr(training_args, key, value)
        else:
            raise AttributeError(f"TrainingArguments has no attribute '{key}'")


if __name__ == "__main__":
    train()
    # train_with_cv()
    pass
