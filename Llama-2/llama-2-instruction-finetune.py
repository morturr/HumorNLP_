# !pip install -q huggingface_hub
# !pip install -q -U trl transformers accelerate peft
# !pip install -q -U datasets bitsandbytes einops wandb

# Uncomment to install new features that support latest models like Llama 2
# !pip install git+https://github.com/huggingface/peft.git
# !pip install git+https://github.com/huggingface/transformers.git

# When prompted, paste the HF access token you created earlier.
import argparse

from typing import List

from huggingface_hub import notebook_login
from huggingface_hub import HfFolder

# notebook_login()

# from datasets import load_dataset
import sys

sys.path.append('../')
sys.path.append('../../')
from FlanT5.data_loader import load_instruction_dataset
from FlanT5.classify_and_evaluate import create_report
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

from sklearn.metrics import (
    classification_report, precision_recall_fscore_support,
)

import os

from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--train_dataset", dest="train_dataset", type=str, required=True)
parser.add_argument("--eval_datasets", dest="eval_datasets", type=str,
                    nargs='+', default=['amazon', 'dadjokes', 'headlines',
                                        'one_liners', 'yelp_reviews'])
parser.add_argument("--instruction_version", dest="instruction_version", type=int, default=-1)
parser.add_argument("--train_batch_size", dest="train_batch_size", type=int, default=4)
parser.add_argument("--inference_batch_size", dest="inference_batch_size", type=int, default=4)
parser.add_argument("--max_steps", dest="max_steps", type=int, default=150)
parser.add_argument("--data_percent", dest="data_percent", type=float, default=100)
parser.add_argument("--max_seq_length", dest="max_seq_length", type=int, default=512)
parser.add_argument("--max_new_tokens", dest="max_new_tokens", type=int, default=5)
parser.add_argument("--task", dest="task", type=str, default='TRAIN')
parser.add_argument("--eval_model_name", dest="eval_model_name", type=str, default='')
args = parser.parse_args()

# dataset_name = args.train_dataset

dataset = load_instruction_dataset(args.train_dataset, instruction_version=args.instruction_version,
                                   percent=args.data_percent)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

device_map = {"": 0}

hf_access_token = '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'


def cuda_memory_status():
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    free_memory = reserved - allocated  # free inside reserved
    print('&& Cuda Memory Info &&')
    print(f'total memory = {total}')
    print(f'reserved memory = {reserved}')
    print(f'allocated memory = {allocated}')
    print(f'free memory = {free_memory}')


def print_run_info(**kwargs):
    info_msgs = ['****',
                 'Trying only yelp with more epochs (200) because the results weren\'t good',
                 'Also increase max seq length to 1024',
                 ]

    for key, value in kwargs.items():
        info_msgs.append(f'{key}: {value}')
    info_msgs.append('****')

    print('\n'.join(info_msgs))


# cuda_memory_status()

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


# def train_llama_versions():
#     global dataset, dataset_name
#
#     NUM_OF_VERSIONS = 6
#
#     # for version_id in range(NUM_OF_VERSIONS):
#     version_id = 2
#     dataset = load_instruction_dataset(dataset_name, percent=0.07, instruction_version=version_id)
#     output_dir = f"Llama-2-7b-hf-{dataset_name}-ver-{version_id}-" \
#                  f"{datetime.now().date()}"
#     train_llama(output_dir)
#
#     # Empty cache after each model training
#     torch.cuda.empty_cache()

def evaluate_llama():
    if not args.eval_model_name:
        raise ValueError('Please provide a model name to evaluate')
    if args.train_dataset not in args.eval_model_name:
        raise ValueError('Please provide a model name that was trained on the same dataset')
    print('Evaluated model: ', args.eval_model_name)
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.eval_model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        # use_auth_token=True,
        token=hf_access_token,
        cache_dir="/cs/labs/dshahaf/mortur/HumorNLP_/Llama-2/Models/",
        # # TODO: remove this also:
        # config=config,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.eval_model_name,
                                              trust_remote_code=True,
                                              token=hf_access_token,
                                              cache_dir="/cs/labs/dshahaf/mortur/HumorNLP_/Llama-2/Models/")
    # Change padding side to left for inference
    tokenizer.padding_side = "left"
    assert tokenizer.padding_side == "left"

    for eval_dataset_name in args.eval_datasets:
        curr_dataset = load_instruction_dataset(eval_dataset_name, instruction_version=args.instruction_version,
                                                percent=args.data_percent)
        split = 'test'

        print(f'@@@ Evaluating on {eval_dataset_name} dataset. Examples count = {len(curr_dataset[split])} @@@')

        # if split is train, remove the answer of output
        def remove_response(row):
            # output = 'Yes' if row['label'] == 1 else 'No'
            # row['instruction'] = row['instruction'] + output
            instruction = row['instruction']
            terminator = '### Output:\n' if '### Output:\n' in instruction else '### Response:\n'
            # row['instruction'] = instruction[:instruction.index('### Output: ') + len('### Output: ')]
            row['instruction'] = instruction[:instruction.index(terminator) + len(terminator)]
            return row

        def get_response(row):
            # output = 'Yes' if row['label'] == 1 else 'No'
            # row['instruction'] = row['instruction'] + output
            # instruction = row['instruction']
            # terminator = '### Output:\n' if '### Output:\n' in instruction else '### Response:\n'
            # # row['instruction'] = instruction[:instruction.index('### Output: ') + len('### Output: ')]
            row['text label'] = 'Yes' if row['label'] == 1 else 'No'
            # row['instruction'] = instruction[instruction.index(terminator) + len(terminator):]
            return row

        def get_prediction(response):
            response_idx = response.index('### Response:\n') + len('### Response:\n')
            response = response[response_idx:]
            if 'Yes' in response and 'No' in response:
                return 'Illegal'
            if 'Yes' in response:
                return 'Yes'
            if 'No' in response:
                return 'No'
            return 'Illegal'

        true_labels = curr_dataset[split].map(get_response)

        if split == 'train':
            curr_dataset[split] = curr_dataset[split].map(remove_response)

        true_labels = [true_labels[i]['text label'] for i in range(len(true_labels))]
        texts = [curr_dataset[split][i]['instruction'] for i in range(len(curr_dataset[split]))]

        inference_batch_size = args.inference_batch_size
        all_responses = []
        prediction_list, label_list = [], []

        for i in range(0, len(texts), inference_batch_size):
            batch_texts = texts[i: i + inference_batch_size]
            batch_labels = true_labels[i: i + inference_batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to("cuda")
            outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"),
                                     attention_mask=inputs["attention_mask"],
                                     max_new_tokens=args.max_new_tokens,
                                     pad_token_id=tokenizer.eos_token_id)

            responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            all_responses.extend(responses)
            batch_predictions = [get_prediction(response) for response in responses]
            prediction_list.extend(batch_predictions)
            label_list.extend(batch_labels)

        prediction_list_int = [1 if prediction == 'Yes' else
                               0 if prediction == 'No'
                               else -1
                               for prediction in prediction_list]

        label_list_int = [1 if prediction == 'Yes' else
                          0 if prediction == 'No'
                          else -1
                          for prediction in label_list]

        # Remove illegal predictions
        illegal_indices = [i for i, val in enumerate(prediction_list_int) if val == -1]
        if illegal_indices:
            print(f'Illegal predictions found in {len(illegal_indices)} examples')
            for index in illegal_indices[::-1]:
                prediction_list_int.pop(index)
                label_list_int.pop(index)

        model_name_idx = len(args.eval_model_name) - args.eval_model_name[::-1].index('/')
        only_model_name = args.eval_model_name[model_name_idx:]

        if os.path.exists('../Results'):
            output_dir = f'../Results/{only_model_name}'
        elif os.path.exists('Results'):
            output_dir = f'Results/{only_model_name}'
        else:
            output_dir = f'{only_model_name}'
        os.makedirs(output_dir, exist_ok=True)

        run_args = {'train_dataset': args.train_dataset,
                    'evaluate_dataset': eval_dataset_name,
                    'model_name': only_model_name,
                    'save_dir': output_dir}

        create_report(label_list_int, prediction_list_int, run_args, pos_label=1)

        curr_dataset[split] = curr_dataset[split].add_column("prediction", prediction_list)
        curr_dataset[split].to_csv(f'{output_dir}/{eval_dataset_name}_predictions.csv')

        with open(f'{output_dir}/instruction_results.txt', 'a') as f:
            f.write(f'@@@ {eval_dataset_name} results')
            for i in range(len(all_responses)):
                f.write(
                    f'*** {i} ***\ntext true response:\n{true_labels[i]}\nmodel output:\n{all_responses[i]}\n\n')
                # print(f'*** {i} ***\ntext true response:\n{true_labels[i]}\nmodel output:\n{all_responses[i]}\n\n')


def train_llama(output_dir=None):
    base_model_name = "meta-llama/Llama-2-7b-hf"
    print(f'^^^^  Training {base_model_name} on {args.train_dataset} ^^^^')
    print(f'^^^ output dir = {output_dir} ^^^')
    train_args = {'num_of_epochs': 20,
                  'max_steps': args.max_steps,
                  'train_dataset': args.train_dataset,
                  'train_size': len(dataset["train"]),
                  'max_new_tokens': 5,
                  'train_batch_size': args.train_batch_size,
                  'inference_batch_size': args.inference_batch_size,
                  'max_seq_length': args.max_seq_length,
                  }

    print_run_info(**train_args)

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
        output_dir = f"../Models/{base_model_name.split('/')[1]}-{dataset_name}-" \
                     f"{datetime.now().date()}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_args['train_batch_size'],
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=train_args['max_steps'],
        report_to='none',
        num_train_epochs=train_args['num_of_epochs'],
    )

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
    # Change padding side to right for training
    tokenizer.padding_side = "right"
    assert tokenizer.padding_side == "right"

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset['train'],
        peft_config=peft_config,
        dataset_text_field="instruction",
        max_seq_length=train_args['max_seq_length'],
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

    # Change padding side to left for inference
    tokenizer.padding_side = "left"
    assert tokenizer.padding_side == "left"

    # for split in ['train', 'test']:
    for split in ['test']:
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

        def get_response(row):
            # output = 'Yes' if row['label'] == 1 else 'No'
            # row['instruction'] = row['instruction'] + output
            # instruction = row['instruction']
            # terminator = '### Output:\n' if '### Output:\n' in instruction else '### Response:\n'
            # # row['instruction'] = instruction[:instruction.index('### Output: ') + len('### Output: ')]
            row['text label'] = 'Yes' if row['label'] == 1 else 'No'
            # row['instruction'] = instruction[instruction.index(terminator) + len(terminator):]
            return row

        def get_prediction(response):
            response_idx = response.index('### Response:\n') + len('### Response:\n')
            response = response[response_idx:]
            if 'Yes' in response and 'No' in response:
                return 'Illegal'
            if 'Yes' in response:
                return 'Yes'
            if 'No' in response:
                return 'No'
            return 'Illegal'

        true_labels = dataset[split].map(get_response)

        if split == 'train':
            dataset[split] = dataset[split].map(remove_response)

        true_labels = [true_labels[i]['text label'] for i in range(len(true_labels))]
        texts = [dataset[split][i]['instruction'] for i in range(len(dataset[split]))]

        inference_batch_size = train_args['inference_batch_size']
        all_responses = []
        prediction_list, label_list = [], []

        for i in range(0, len(texts), inference_batch_size):
            batch_texts = texts[i: i + inference_batch_size]
            batch_labels = true_labels[i: i + inference_batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to("cuda")
            outputs = trainer.model.generate(input_ids=inputs["input_ids"].to("cuda"),
                                             attention_mask=inputs["attention_mask"],
                                             max_new_tokens=train_args['max_new_tokens'],
                                             pad_token_id=tokenizer.eos_token_id)

            responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            all_responses.extend(responses)
            batch_predictions = [get_prediction(response) for response in responses]
            prediction_list.extend(batch_predictions)
            label_list.extend(batch_labels)

        prediction_list_int = [1 if prediction == 'Yes' else
                               0 if prediction == 'No'
                               else -1
                               for prediction in prediction_list]

        label_list_int = [1 if prediction == 'Yes' else
                          0 if prediction == 'No'
                          else -1
                          for prediction in label_list]

        # Remove illegal predictions
        illegal_indices = [i for i, val in enumerate(prediction_list_int) if val == -1]
        if illegal_indices:
            print(f'Illegal predictions found in {len(illegal_indices)} examples')
            for index in illegal_indices[::-1]:
                print(f'Removing illegal prediction #{index}: {prediction_list[index]}')
                prediction_list_int.pop(index)
                label_list_int.pop(index)

        # TODO Mor: temporary model name
        model_name = f'{base_model_name}-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-{dataset_name}'
        run_args = {'train_dataset': dataset_name,
                    'evaluate_dataset': dataset_name,
                    'model_name': model_name}

        create_report(label_list_int, prediction_list_int, run_args, pos_label=1)

        with open(f'{output_dir}/instruction_results.txt', 'a') as f:
            f.write(f'@@@ {split} split results')
            for i in range(len(all_responses)):
                f.write(
                    f'*** {i} ***\ntext true response:\n{true_labels[i]}\nmodel output:\n{all_responses[i]}\n\n')
                print(f'*** {i} ***\ntext true response:\n{true_labels[i]}\nmodel output:\n{all_responses[i]}\n\n')


#
# def load_llama():
#     id2yesno = {0: "No", 1: "Yes"}
#     yesno2id = {label: id for id, label in id2yesno.items()}
#     # trained_model_name = "morturr/Llama-2-7b-hf-headlines-2024-07-11"
#     trained_model_name = "morturr/Llama-2-7b-hf-headlines-2024-07-16"
#
#     device_map = {"": 0}
#
#     hf_access_token = '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'
#
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float16,
#     )
#
#     trained_model = AutoModelForCausalLM.from_pretrained(
#         trained_model_name,
#         quantization_config=bnb_config,
#         device_map=device_map,
#         trust_remote_code=True,
#         # use_auth_token=True,
#         token=hf_access_token,
#         cache_dir="/cs/labs/dshahaf/mortur/HumorNLP_/Llama-2/Cache/",
#         # config=AutoConfig.from_pretrained(
#         #     trained_model_name, num_labels=len(id2yesno), id2label=id2yesno, label2id=yesno2id
#     )
#
#     trained_model.config.use_cache = False
#
#     tokenizer = AutoTokenizer.from_pretrained(trained_model_name,
#                                               trust_remote_code=True,
#                                               token=hf_access_token,
#                                               cache_dir="/cs/labs/dshahaf/mortur/HumorNLP_/Llama-2/Cache/")
#     tokenizer.pad_token = tokenizer.eos_token
#
#     pass
#
#     predictions_list, labels_list = [], []
#     batch_size = 8
#     batch_texts = dataset["train"]["text"][:batch_size]
#     batch_labels = dataset["train"]["label"][:batch_size]
#
#     texts_to_classify = batch_texts
#     inputs = tokenizer(
#         texts_to_classify,
#         return_tensors="pt",
#         max_length=512,
#         truncation=True,
#         padding=True,
#     )
#     inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
#
#     with torch.no_grad():
#         outputs = trained_model(**inputs)
#
#     # Process the outputs to get the probability distribution
#     logits = outputs.logits
#     probs = torch.nn.functional.softmax(logits, dim=-1)
#
#     # Get the top class and the corresponding probability (certainty) for each input text
#     confidences, predicted_classes = torch.max(probs, dim=1)
#     predicted_classes = (
#         predicted_classes.cpu().numpy()
#     )  # Move to CPU for numpy conversion if needed
#     confidences = confidences.cpu().numpy()  # Same here
#
#     # Map predicted class IDs to labels
#     print('*** predicted classes ***')
#     print(f'type = {type(predicted_classes)}\n value = {predicted_classes}')
#     predicted_labels = [id2yesno[class_id] for class_id in predicted_classes]
#
#     # Zip together the predicted labels and confidences and convert to a list of tuples
#     batch_predictions = list(zip(predicted_labels, confidences))
#
#     predictions_list.extend(batch_predictions)
#     labels_list.extend([id2yesno[label_id] for label_id in batch_labels])
#
#     predictions_list = [pair[0] for pair in predictions_list]
#     print(f'prediction list:\n {predictions_list}')
#     print(f'labels list:\n {labels_list}')
#     report = classification_report(labels_list, predictions_list)
#     print(report)
#

if __name__ == '__main__':
    if args.task == 'TRAIN':
        train_llama()

    if args.task == 'EVAL':
        evaluate_llama()
    # load_llama()
    # train_llama_versions()
