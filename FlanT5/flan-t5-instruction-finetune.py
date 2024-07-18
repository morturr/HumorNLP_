# !pip install -q huggingface_hub
# !pip install -q -U trl transformers accelerate peft
# !pip install -q -U datasets bitsandbytes einops wandb

# Uncomment to install new features that support latest models like Llama 2
# !pip install git+https://github.com/huggingface/peft.git
# !pip install git+https://github.com/huggingface/transformers.git

# When prompted, paste the HF access token you created earlier.

from huggingface_hub import notebook_login
from huggingface_hub import HfFolder

# notebook_login()

# from datasets import load_dataset
import sys
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoConfig,
    LlamaConfig,
)


from peft import LoraConfig, TaskType
from trl import SFTTrainer
from data_loader import load_instruction_dataset

from sklearn.metrics import (
    classification_report, precision_recall_fscore_support,
)

import os

from datetime import datetime


dataset_name = "headlines"
# dataset = load_dataset(dataset_name, split="train")

if len(sys.argv) > 1:
    instruction_version = sys.argv[1]
else:
    instruction_version = None
dataset = load_instruction_dataset(dataset_name, percent=0.1, instruction_version=instruction_version)


# def train_flan():
#     base_model_name = "google/flan-t5-base"
#     # base_model_name = "meta-llama/Llama-2-7b-hf"
#
#     # bnb_config = BitsAndBytesConfig(
#     #     load_in_4bit=True,
#     #     bnb_4bit_quant_type="nf4",
#     #     bnb_4bit_compute_dtype=torch.float16,
#     # )
#
#     device_map = {"": 0}
#
#     # Used with Llama
#     # base_model = AutoModelForCausalLM.from_pretrained(
#     base_model = AutoModelForSequenceClassification.from_pretrained(
#         base_model_name,
#         # quantization_config=bnb_config,
#         device_map=device_map,
#         trust_remote_code=True,
#     )
#     base_model.config.use_cache = False
#
#     # More info: https://github.com/huggingface/transformers/pull/24906
#     # Use for Llama
#     # base_model.config.pretraining_tp = 1
#
#     peft_config = LoraConfig(
#         lora_alpha=16,
#         lora_dropout=0.1,
#         r=64,
#         bias="none",
#         # task_type="CAUSAL_LM",
#         ## I added the rows below:
#         target_modules=["q", "v"],
#         task_type=TaskType.SEQ_2_SEQ_LM  # FLAN-T5
#     )
#
#     # This one is from different blog (talking about LoRA with flan-T5)
#     lora_config = LoraConfig(
#         r=32,  # Rank
#         lora_alpha=32,
#         target_modules=["q", "v"],
#         lora_dropout=0.1,
#         bias="lora_only",
#         task_type=TaskType.SEQ_2_SEQ_LM  # FLAN-T5
#     )
#
#     tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
#     # tokenizer.pad_token = tokenizer.eos_token
#
#     output_dir = "./results"
#
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         # per_device_train_batch_size=4,
#         per_device_train_batch_size=8,
#         # gradient_accumulation_steps=4,
#         learning_rate=2e-4,
#         logging_steps=10,
#         max_steps=500,
#         hub_token=HfFolder.get_token(),
#         report_to="none",
#     )
#
#     max_seq_length = 1024
#     # max_seq_length = 512
#
#     trainer = SFTTrainer(
#         model=base_model,
#         train_dataset=dataset['train'],
#         peft_config=lora_config,
#         # peft_config=peft_config,
#         dataset_text_field="instruction",
#         max_seq_length=max_seq_length,
#         tokenizer=tokenizer,
#         args=training_args,
#     )
#
#     trainer.train()
#
#
#     output_dir = os.path.join(output_dir, "final_checkpoint")
#     trainer.model.save_pretrained(output_dir)

def train_llama_versions():
    global dataset, dataset_name

    NUM_OF_VERSIONS = 6

    for version_id in range(NUM_OF_VERSIONS):
        dataset = load_instruction_dataset(dataset_name, percent=0.07, instruction_version=version_id)
        output_dir = f"Llama-2-7b-hf-{dataset_name}-ver-{version_id}-" \
                                f"{datetime.now().date()}"
        # train_llama(output_dir)


def train_llama(output_dir=None):
    base_model_name = "meta-llama/Llama-2-7b-hf"
    print(f'^^^^  Training {base_model_name} on {dataset_name} ^^^^')
    num_of_epochs = 10
    print(f'^^^ epochs = {num_of_epochs} train size = {len(dataset["train"])}')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    device_map = {"": 0}

    hf_access_token = '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'

    # # TODO: remove this. Trying to train Llama in a similar manner to Flan (using labels)
    # id2yesno = {0: "No", 1: "Yes"}
    # yesno2id = {label: id for id, label in id2yesno.items()}
    # config = LlamaConfig.from_pretrained(
    #     base_model_name, num_labels=len(yesno2id), id2label=id2yesno, label2id=yesno2id,
    #     token=hf_access_token
    # )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        # use_auth_token=True,
        token=hf_access_token,
        cache_dir="/cs/labs/dshahaf/mortur/HumorNLP_/Llama-2/Models/",
        # # TODO: remove this also:
        # config=config,
    )
    base_model.config.use_cache = False

    # More info: https://github.com/huggingface/transformers/pull/24906
    base_model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name,
                                              trust_remote_code=True,
                                              token=hf_access_token,
                                              cache_dir="/cs/labs/dshahaf/mortur/HumorNLP_/Llama-2/Models/")
    tokenizer.pad_token = tokenizer.eos_token

    if not output_dir:
        output_dir = f"{base_model_name.split('/')[1]}-{dataset_name}-" \
                                f"{datetime.now().date()}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=500,
        report_to='none',
        num_train_epochs=num_of_epochs,
    )

    max_seq_length = 512

    # # TODO: for llama train with labels. remove this later
    # def compute_metrics(eval_pred) -> dict:
    #     """Compute metrics for evaluation"""
    #     logits, labels = eval_pred
    #     if isinstance(
    #             logits, tuple
    #     ):  # if the model also returns hidden_states or attentions
    #         logits = logits[0]
    #     predictions = np.argmax(logits, axis=-1)
    #     precision, recall, f1, _ = precision_recall_fscore_support(
    #         labels, predictions, average="binary"
    #     )
    #     return {"precision": precision, "recall": recall, "f1": f1}

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset['train'],
        peft_config=peft_config,
        dataset_text_field="instruction",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        # # TODO: Llama with labels, to remove
        # compute_metrics=compute_metrics,
    )

    trainer.train()

    # output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    trainer.create_model_card()
    trainer.push_to_hub(token=hf_access_token)

    # predict on the test set
    # model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map=device_map, torch_dtype=torch.bfloat16)
    for split in ['train', 'test']:
        print(f'@@@ Evaluating on {split} split. Examples count = {len(dataset[split])} @@@')

        # if split is train, remove the answer of output
        def remove_response(row):
            # output = 'Yes' if row['label'] == 1 else 'No'
            # row['instruction'] = row['instruction'] + output
            instruction = row['instruction']
            terminator = '### Output:\n' if '### Output:\n' in instruction else '### Response:\n'
            # row['instruction'] = instruction[:instruction.index('### Output: ') + len('### Output: ')]
            row['instruction'] = instruction[:instruction.index(terminator) + len(terminator)]
            return row

        if split == 'train':
            dataset[split] = dataset[split].map(remove_response)

        texts = [dataset[split][i]['instruction'] for i in range(len(dataset[split]))]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
        outputs = base_model.generate(input_ids=inputs["input_ids"].to("cuda"), attention_mask=inputs["attention_mask"],
                                 max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

        responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        with open(f'{output_dir}/instruction_results.txt', 'a') as f:
            f.write(f'@@@ {split} split results')
            for i in range(len(responses)):
                f.write(f'*** {i} ***\ntext:\n{texts[i]}\nmodel response:\n{responses[i]}\n\n')
                print(f'*** {i} ***\ntext:\n{texts[i]}\nmodel response:\n{responses[i]}\n\n')

        pass


def load_llama():
    id2yesno = {0: "No", 1: "Yes"}
    yesno2id = {label: id for id, label in id2yesno.items()}
    # trained_model_name = "morturr/Llama-2-7b-hf-headlines-2024-07-11"
    trained_model_name = "morturr/Llama-2-7b-hf-headlines-2024-07-16"

    device_map = {"": 0}

    hf_access_token = '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'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    trained_model = AutoModelForCausalLM.from_pretrained(
        trained_model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        # use_auth_token=True,
        token=hf_access_token,
        cache_dir="/cs/labs/dshahaf/mortur/HumorNLP_/Llama-2/Cache/",
        # config=AutoConfig.from_pretrained(
        #     trained_model_name, num_labels=len(id2yesno), id2label=id2yesno, label2id=yesno2id
        )

    trained_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(trained_model_name,
                                              trust_remote_code=True,
                                              token=hf_access_token,
                                              cache_dir="/cs/labs/dshahaf/mortur/HumorNLP_/Llama-2/Cache/")
    tokenizer.pad_token = tokenizer.eos_token

    pass

    predictions_list, labels_list = [], []
    batch_size = 8
    batch_texts = dataset["train"]["text"][:batch_size]
    batch_labels = dataset["train"]["label"][:batch_size]

    texts_to_classify = batch_texts
    inputs = tokenizer(
        texts_to_classify,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = trained_model(**inputs)

    # Process the outputs to get the probability distribution
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)


    # Get the top class and the corresponding probability (certainty) for each input text
    confidences, predicted_classes = torch.max(probs, dim=1)
    predicted_classes = (
        predicted_classes.cpu().numpy()
    )  # Move to CPU for numpy conversion if needed
    confidences = confidences.cpu().numpy()  # Same here

    # Map predicted class IDs to labels
    print('*** predicted classes ***')
    print(f'type = {type(predicted_classes)}\n value = {predicted_classes}')
    predicted_labels = [id2yesno[class_id] for class_id in predicted_classes]

    # Zip together the predicted labels and confidences and convert to a list of tuples
    batch_predictions = list(zip(predicted_labels, confidences))

    predictions_list.extend(batch_predictions)
    labels_list.extend([id2yesno[label_id] for label_id in batch_labels])

    predictions_list = [pair[0] for pair in predictions_list]
    print(f'prediction list:\n {predictions_list}')
    print(f'labels list:\n {labels_list}')
    report = classification_report(labels_list, predictions_list)
    print(report)


if __name__ == '__main__':
    # train_llama()
    # load_llama()
    train_llama_versions()
