import argparse
import csv

import psutil
from huggingface_hub import HfFolder
from datasets import Dataset, DatasetDict

import random
from itertools import product
import time
import sys

sys.path.append('../')
sys.path.append('../../')
from FlanT5.data_loader import (
    load_dataset,
    load_cv_dataset,
    load_dataset_domain_classification,
)

from FlanT5.classify_and_evaluate import create_report
import torch
import pandas as pd

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)

from peft import LoraConfig
from trl import SFTTrainer

import os
import GPUtil

cwd = os.getcwd()
current_dir, dir_above = cwd.split('/')[-1], cwd.split('/')[-2]
# BASE_MODEL_NAME = 'Llama-2'
# base_model_name_default = "meta-llama/Llama-2-7b-hf"
BASE_MODEL_NAME = 'Mistral'
base_model_name_default = "mistralai/Mistral-7B-v0.1"


from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--train_dataset", dest="train_dataset", type=str)  # , required=True)
parser.add_argument("--eval_datasets", dest="eval_datasets", type=str,
                    nargs='+', default=['amazon', 'dadjokes', 'headlines',
                                        'one_liners'])
parser.add_argument("--instruction_version", dest="instruction_version", type=int, default=0)
parser.add_argument("--train_batch_size", dest="train_batch_size", type=int, default=4)
parser.add_argument("--inference_batch_size", dest="inference_batch_size", type=int, default=4)
parser.add_argument("--data_percent", dest="data_percent", type=float, default=1)
parser.add_argument("--test_data_percent", dest="test_data_percent", type=float, default=0.1)
parser.add_argument("--train_size", dest="train_size", type=int, default=None)
parser.add_argument("--max_seq_length", dest="max_seq_length", type=int, default=1024)
parser.add_argument("--max_new_tokens", dest="max_new_tokens", type=int, default=5)
parser.add_argument("--task", dest="task", type=str, default='TRAIN')
parser.add_argument("--eval_model_name", dest="eval_model_name", type=str, default='')
parser.add_argument("--base_model_name", dest="base_model_name", type=str, default=base_model_name_default)

parser.add_argument('--seeds', dest='seeds', type=int, nargs='+', default=[42])
parser.add_argument('--batch_sizes', dest='batch_sizes', type=int, nargs='+', default=[4])
parser.add_argument('--learning_rates', dest='learning_rates', type=float, nargs='+', default=[3e-4, 5e-5, 6e-5])
parser.add_argument('--lora_ranks', dest='lora_ranks', type=int, nargs='+', default=[32, 64, 128])
parser.add_argument('--lora_alpha', dest='lora_alpha', type=float, nargs='+', default=[8, 16, 32])
parser.add_argument('--num_train_epochs', dest='num_train_epochs', type=int, nargs='+', default=[2])

# Leave One Out parameters
parser.add_argument('--param_combination_file', dest='param_combination_file', type=str, default='')
parser.add_argument('--loo_datasets', dest='loo_datasets', type=str, nargs='+',
                    default=['amazon', 'dadjokes', 'headlines', 'one_liners'])
parser.add_argument('--leave_out', dest='leave_out', type=str, default='')
parser.add_argument('--loo_dataset_for_combs', dest='loo_dataset_for_combs', type=str, default='')
parser.add_argument('--loo_comb_of_all_datasets', action=argparse.BooleanOptionalAction)
parser.add_argument('--zero_shot', type=int, default=0)

parser.add_argument('--combined_datasets', dest='combined_datasets', type=str, nargs='+',
                    default=[])

args = parser.parse_args()
# dataset_name = args.train_dataset

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

device_map = {"": 0}

print(f"Using device: {torch.cuda.current_device()}, total devices: {torch.cuda.device_count()}")

hf_access_token = ''
if not hf_access_token:
    print('Please set your HuggingFace access token in the hf_access_token variable.')
    exit()

# Check if we're directly in Llama/Mistral directory
cwd = os.getcwd()
current_dir = cwd.split('/')[-1]
if current_dir == 'Llama-2' or current_dir == 'Mistral':
    REPO_ID_PREFIX = './Models/'
elif current_dir == 'Scripts':
    REPO_ID_PREFIX = '../Models/'
else:
    raise Exception('Please run the script from the Llama-2/Mistral or Scripts directory')


def cuda_memory_status(code_location: str = ''):
    print(code_location)
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory / (2 ** 30)
    reserved = torch.cuda.memory_reserved(0) / (2 ** 30)
    allocated = torch.cuda.memory_allocated(0) / (2 ** 30)
    free_memory = reserved - allocated  # free inside reserved
    print('&& Cuda Memory Info &&')
    print(f'total memory = {total}')
    print(f'reserved memory = {reserved}')
    print(f'allocated memory = {allocated}')
    print(f'free memory = {free_memory}')


def log_memory_usage(code_location: str = ''):
    print(code_location)
    process = psutil.Process()
    print(f"CPU Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    gpus = GPUtil.getGPUs()
    if gpus:
        print(f"GPU Memory: {gpus[0].memoryUsed}MB / {gpus[0].memoryTotal}MB")


def print_run_info(**kwargs):
    info_msgs = ['****',
                 'Trying only yelp with more epochs (200) because the results weren\'t good',
                 'Also increase max seq length to 1024',
                 ]

    for key, value in kwargs.items():
        info_msgs.append(f'{key}: {value}')
    info_msgs.append('****')

    print('\n'.join(info_msgs))


def update_training_arguments(training_args, **kwargs):
    for key, value in kwargs.items():
        if hasattr(training_args, key):
            setattr(training_args, key, value)
        else:
            raise AttributeError(f"TrainingArguments has no attribute '{key}'")

    return training_args


def init_model_wrapper(model_name):
    def init_model():
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            # use_auth_token=True,
            token=hf_access_token,
            cache_dir=f"/cs/labs/dshahaf/mortur/HumorNLP_/{BASE_MODEL_NAME}/Cache/",
            # config=config,
        )
        base_model.config.use_cache = False

        # More info: https://github.com/huggingface/transformers/pull/24906
        base_model.config.pretraining_tp = 1
        return base_model

    return init_model


def get_existing_combinations(df_combs, all_combs):
    selected_combs = []

    for index, row in df_combs.iterrows():
        comb = tuple(row)
        print(comb)
        all_combs.remove(comb)
        selected_combs.append(comb)

    return selected_combs, all_combs


def set_run_params(params, repo_id):
    print(f"Training with hyperparameters: {params}")

    print(f'training {repo_id}')

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=repo_id,
        num_train_epochs=params['num_train_epochs'],
        per_device_train_batch_size=params['per_device_train_batch_size'],
        per_device_eval_batch_size=params['per_device_eval_batch_size'],
        learning_rate=params['learning_rate'],
        seed=params['seed'],
        # hub_model_id=REPOSITORY_ID,
        hub_token=HfFolder.get_token(),
        gradient_accumulation_steps=4,
        logging_steps=10,
        report_to='none',
        gradient_checkpointing=True,
        label_names=['domain_label']
    )

    # training_args = update_training_arguments(training_args, **training_args_update)
    lora_args = {'rank': params['lora_rank'], 'alpha': params['lora_alpha']}

    return training_args, lora_args


def evaluate(eval_params=None, additional_run_args={}, is_cross_eval=False,
             cv_split_num=None):
    with torch.no_grad():
        if eval_params:
            model = eval_params['model']
            tokenizer = eval_params['tokenizer']
            train_dataset_name = eval_params['train_dataset_name']
            eval_datasets = eval_params['eval_datasets']
            eval_model_name = eval_params['model_location']

        else:
            if not args.eval_model_name:
                raise ValueError('Please provide a model name to evaluate')

            if not args.zero_shot:
                if args.train_dataset not in args.eval_model_name:
                    raise ValueError('Please provide a model name that was trained on the same dataset')

            eval_model_name = args.eval_model_name
            train_dataset_name = args.train_dataset
            eval_datasets = args.eval_datasets

            print('Evaluated model: ', args.eval_model_name)
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                args.eval_model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
                # use_auth_token=True,
                token=hf_access_token,
                cache_dir=f"/cs/labs/dshahaf/mortur/HumorNLP_/{BASE_MODEL_NAME}/Cache/",
                # config=config,
            )

            tokenizer = AutoTokenizer.from_pretrained(args.eval_model_name,
                                                      trust_remote_code=True,
                                                      token=hf_access_token,
                                                      cache_dir=f"/cs/labs/dshahaf/mortur/HumorNLP_/{BASE_MODEL_NAME}/Cache/")

        # Change padding side to left for inference
        tokenizer.padding_side = "left"
        assert tokenizer.padding_side == "left"

        if args.zero_shot:
            tokenizer.pad_token = tokenizer.eos_token

        # TODO Mor: my additions, remove it later
        # backup_model = model
        # from peft import PeftModel
        # peft_model = get_peft_model(model, model.peft_config['default'])
        # trainer = SFTTrainer(model=model, dataset_text_field='instruction', max_seq_length=1024,
        # peft_config=model.peft_config['default'])

        # print(f'backup_model == trainer.model: {backup_model == trainer.model}')

        start_eval_time = time.time()

        curr_dataset = load_dataset_domain_classification(datasets_names=args.loo_datasets,
                                                   num_of_samples=5000,
                                                   which_labels='negative')#, data_percent=0.1)

        split = 'test'

        print(f'@@@ Evaluating... Examples count = {len(curr_dataset[split])} @@@')

        def extract_response(response):
            if '### Response:\n' not in response:
                return 'Illegal'
            response_idx = response.index('### Response:\n') + len('### Response:\n')
            extracted_response = response[response_idx:]
            return extracted_response

        def get_prediction(extracted_response):
            domainlabel2id = {name: i for i, name in enumerate(args.loo_datasets)}
            if extracted_response == 'Illegal':
                return 'Illegal'

            found_label, predicted_label = False, 'Illegal'

            for label in domainlabel2id:
                if label in extracted_response:
                    if not found_label:
                        found_label = True
                        predicted_label = label
                    else:
                        # More than one label found, return Illegal
                        return 'Illegal'

            return predicted_label

        id2domainlabel = {i: name for i,name in enumerate(args.loo_datasets)}
        domainlabel2id = {name: i for i, name in enumerate(args.loo_datasets)}

        true_labels = [id2domainlabel[id] for id in curr_dataset[split]['domain_label']]


        # true_labels = [true_labels[i]['text label'] for i in range(len(true_labels))]
        texts = [curr_dataset[split][i]['instruction'] for i in range(len(curr_dataset[split]))]

        inference_batch_size = args.inference_batch_size
        all_responses, all_extracted_responses = [], []
        prediction_list, label_list = [], []

        for i in range(0, len(texts), inference_batch_size):
            batch_texts = texts[i: i + inference_batch_size]
            batch_labels = true_labels[i: i + inference_batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to("cuda")
            # TODO Mor: remove it later
            # outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"),
            #                          attention_mask=inputs["attention_mask"],
            #                          max_new_tokens=args.max_new_tokens,
            #                          pad_token_id=tokenizer.eos_token_id)

            outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"),
                                          attention_mask=inputs["attention_mask"],
                                          max_new_tokens=5,
                                          pad_token_id=tokenizer.eos_token_id)

            responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            all_responses.extend(responses)
            extracted_responses = [extract_response(response) for response in responses]
            all_extracted_responses.extend(extracted_responses)
            batch_predictions = [get_prediction(ext_response) for ext_response in extracted_responses]
            prediction_list.extend(batch_predictions)
            label_list.extend(batch_labels)

        prediction_list_int = [domainlabel2id.get(label, -1)
                               for label in prediction_list]

        label_list_int = [domainlabel2id.get(label, -1)
                          for label in label_list]

        # Remove illegal predictions
        illegal_indices = [i for i, val in enumerate(prediction_list_int) if val == -1]
        if illegal_indices:
            print(f'Illegal predictions found in {len(illegal_indices)} examples')
            print(f'Illegal Responses:')
            for index in illegal_indices[::-1]:
                print(f'Index: {index}, Response: {all_extracted_responses[index]}')
                prediction_list_int.pop(index)
                label_list_int.pop(index)

        if '/' in eval_model_name:
            model_name_idx = len(eval_model_name) - eval_model_name[::-1].index('/')
            only_model_name = eval_model_name[model_name_idx:]
        else:
            only_model_name = eval_model_name

        if os.path.exists('../Results'):
            output_dir = f'../Results/{only_model_name}'
        elif os.path.exists('Results'):
            output_dir = f'Results/{only_model_name}'
        else:
            output_dir = f'{only_model_name}'
        os.makedirs(output_dir, exist_ok=True)

        run_args = {'train_dataset': 'domain_classification_negative',
                    'evaluate_dataset': 'domain_classification_negative',
                    'model_name': only_model_name,
                    'save_dir': output_dir}

        run_args.update(additional_run_args)

        create_report(label_list_int, prediction_list_int, run_args, pos_label=1, average=None)

        curr_dataset[split] = curr_dataset[split].add_column("prediction", prediction_list)
        curr_dataset[split].to_csv(f'{output_dir}/domain_classification_negative_predictions.csv')
        # Save the extracted responses
        with open(f'{output_dir}/domain_classification_negative_responses.txt', 'w') as f:
            for response in all_extracted_responses:
                f.write(response.strip() + '\n*****\n')

        # with open(f'{output_dir}/instruction_results.txt', 'a') as f:
        #     f.write(f'@@@ {eval_dataset_name} results')
        #     for i in range(len(all_responses)):
        #         f.write(
        #             f'*** {i} ***\ntext true response:\n{true_labels[i]}\nmodel output:\n{all_responses[i]}\n\n')

        end_eval_time = time.time()
        print(f'Evaluation took {(end_eval_time - start_eval_time) / 60} minutes')

    # Clear memory
    print('|||||', 'Evaluation: Clearing memory', '|||||', sep='\n')
    import gc
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def train_domain_classification(output_dir=None, training_args=None, data_dict=None, lora_args=None,
           additional_run_args={}, is_cross_val=False, cv_split_num=None,
           train_dataset_name=args.train_dataset,
           model_name=args.base_model_name, eval_datasets_names=args.eval_datasets):
    print(f'^^^^  Training {model_name} on {train_dataset_name} ^^^^')
    train_start_time = time.time()

    if not output_dir:
        output_dir = f"{REPO_ID_PREFIX}{model_name.split('/')[1]}" \
                     f"-{train_dataset_name}-" \
                     f"seed-{args.seeds[0]}-" \
                     f"{datetime.now().date()}"

    print(f'^^^ output dir = {output_dir} ^^^')

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        # use_auth_token=True,
        token=hf_access_token,
        cache_dir=f"/cs/labs/dshahaf/mortur/HumorNLP_/{BASE_MODEL_NAME}/Cache2/",
        # config=config,
    )
    base_model.config.use_cache = False

    # More info: https://github.com/huggingface/transformers/pull/24906
    base_model.config.pretraining_tp = 1

    if not training_args:
        assert len(args.seeds) == 1
        assert len(args.learning_rates) == 1
        assert len(args.batch_sizes) == 1
        assert len(args.lora_ranks) == 1
        assert len(args.lora_alpha) == 1
        assert len(args.num_train_epochs) == 1

        print('** Loading training arguments from command line arguments **')
        # Set the run parameters
        current_params = {
            'seed': args.seeds[0],
            'learning_rate': args.learning_rates[0],
            'per_device_train_batch_size': args.batch_sizes[0],
            'per_device_eval_batch_size': args.batch_sizes[0],
            'lora_rank': args.lora_ranks[0],
            'lora_alpha': args.lora_alpha[0],
            'num_train_epochs': args.num_train_epochs[0],
        }

        training_args, lora_args = set_run_params(current_params, output_dir)

    peft_config = LoraConfig(
        lora_alpha=16 if not lora_args else lora_args['alpha'],
        lora_dropout=0.1,
        r=64 if not lora_args else lora_args['rank'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True,
                                              token=hf_access_token,
                                              cache_dir=f"/cs/labs/dshahaf/mortur/HumorNLP_/{BASE_MODEL_NAME}/Cache/")
    tokenizer.pad_token = tokenizer.eos_token

    # Change padding side to right for training
    tokenizer.padding_side = "right"
    assert tokenizer.padding_side == "right"

    if data_dict:
        train_dataset = data_dict['train']
    else:
        train_dataset = load_dataset_domain_classification(datasets_names=args.loo_datasets,
                                                           num_of_samples=5000,
                                                           which_labels='negative', data_percent=None)['train']

    # train_dataset = dataset['train'] if data_dict is None else data_dict['train']
    print('$$$ Training on dataset with size:', len(train_dataset), '$$$')
    # TODO Mor: I change it to None to see how it affects the training
    args.max_seq_length = None
    print(f'&&& max seq length = {args.max_seq_length} &&&')

    trainer = SFTTrainer(
        model=base_model,
        # model_init=init_model_wrapper(args.base_model_name),
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="instruction",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()

    train_end_time = time.time()
    print('&&&&&&&&&&&&&&&&&&&&&')
    print(f'Training took {(train_end_time - train_start_time) / 60} minutes')
    print('&&&&&&&&&&&&&&&&&&&&&')

    # output_dir = os.path.join(output_dir, "final_checkpoint")

    trainer.model.save_pretrained(output_dir)
    trainer.create_model_card()
    trainer.push_to_hub(token=hf_access_token)

    eval_params = {}
    eval_params['model'] = trainer.model
    eval_params['tokenizer'] = tokenizer
    eval_params['train_dataset_name'] = train_dataset_name
    eval_params['eval_datasets'] = eval_datasets_names
    eval_params['model_location'] = output_dir

    eval_start_time = time.time()
    evaluate(eval_params, additional_run_args, is_cross_val, cv_split_num)
    eval_end_time = time.time()
    print('@@@@@@@@@@@@@@@')
    print(f'Evaluation took {(eval_end_time - eval_start_time) / 60} minutes')
    print('@@@@@@@@@@@@@@@')

    # Clear memory
    print('|||||', 'Clearing memory', '|||||', sep='\n')
    import gc
    trainer.model.to('cpu')
    del base_model, trainer.model, trainer, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    output_dir = f"{REPO_ID_PREFIX}{base_model_name_default.split('/')[1]}" \
                     f"-DomainClassification-Negative-" \
                     f"seed-{args.seeds[0]}-" \
                     f"{datetime.now().date()}"
    if args.task == 'TRAIN_DOMAIN_CLASSIFICATION':
        train_domain_classification(output_dir=output_dir)

    if args.task == 'EVAL':
        evaluate()

    if args.task == '':
        print('Nothing were given to do. Please provide a task to do.')
