# !pip install -q huggingface_hub
# !pip install -q -U trl transformers accelerate peft
# !pip install -q -U datasets bitsandbytes einops wandb

# Uncomment to install new features that support latest models like Llama 2
# !pip install git+https://github.com/huggingface/peft.git
# !pip install git+https://github.com/huggingface/transformers.git

# When prompted, paste the HF access token you created earlier.
import argparse
import csv

from typing import List

import psutil
from huggingface_hub import notebook_login
from huggingface_hub import HfFolder
from datasets import Dataset, DatasetDict

# notebook_login()

import random
from itertools import product
import time
# from datasets import load_dataset
import sys

sys.path.append('../')
sys.path.append('../../')
from FlanT5.data_loader import (
    load_dataset,
    load_cv_dataset,
    load_LOO_datasets,
    load_current_LOO,
    load_combined_dataset
)

from FlanT5.classify_and_evaluate import create_report
import torch
import numpy as np
import pandas as pd

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoConfig,
    LlamaConfig,
)

from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer

from sklearn.metrics import (
    classification_report, precision_recall_fscore_support,
)

import os
import GPUtil


cwd = os.getcwd()
current_dir, dir_above = cwd.split('/')[-1], cwd.split('/')[-2]
if current_dir == 'Llama-2' or dir_above == 'Llama-2':
    BASE_MODEL_NAME = 'Llama-2'
    base_model_name_default = "meta-llama/Llama-2-7b-hf"
elif current_dir == 'Mistral' or dir_above == 'Mistral':
    BASE_MODEL_NAME = 'Mistral'
    base_model_name_default = "mistralai/Mistral-7B-v0.1"
else:
    raise Exception('Please run the script from the Llama-2 or Mistral directory')

from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--train_dataset", dest="train_dataset", type=str)#, required=True)
parser.add_argument("--eval_datasets", dest="eval_datasets", type=str,
                    nargs='+', default=['amazon', 'dadjokes', 'headlines',
                                        'one_liners', 'yelp_reviews'])
parser.add_argument("--instruction_version", dest="instruction_version", type=int, default=0)
parser.add_argument("--train_batch_size", dest="train_batch_size", type=int, default=4)
parser.add_argument("--inference_batch_size", dest="inference_batch_size", type=int, default=4)
parser.add_argument("--data_percent", dest="data_percent", type=float, default=1)
parser.add_argument("--test_data_percent", dest="test_data_percent", type=float, default=0.1)
parser.add_argument("--max_seq_length", dest="max_seq_length", type=int, default=1024)
parser.add_argument("--max_new_tokens", dest="max_new_tokens", type=int, default=5)
parser.add_argument("--task", dest="task", type=str, default='TRAIN')
parser.add_argument("--eval_model_name", dest="eval_model_name", type=str, default='')
parser.add_argument("--base_model_name", dest="base_model_name", type=str, default="meta-llama/Llama-2-7b-hf")

parser.add_argument('--seeds', dest='seeds', type=int, nargs='+', default=[42])
parser.add_argument('--num_steps', dest='num_steps', type=int, nargs='+', default=[150, 200])
parser.add_argument('--batch_sizes', dest='batch_sizes', type=int, nargs='+', default=[4])
parser.add_argument('--learning_rates', dest='learning_rates', type=float, nargs='+', default=[3e-4, 5e-5, 1e-5, 5e-6])
parser.add_argument('--lora_ranks', dest='lora_ranks', type=int, nargs='+', default=[32, 64, 128])
parser.add_argument('--lora_alpha', dest='lora_alpha', type=float, nargs='+', default=[8, 16, 32, 64])

# Leave One Out parameters
parser.add_argument('--param_combination_file', dest='param_combination_file', type=str, default='')
parser.add_argument('--loo_datasets', dest='loo_datasets', type=str, nargs='+',
                    default=['amazon', 'dadjokes', 'headlines', 'one_liners', 'yelp_reviews'])
parser.add_argument('--leave_out', dest='leave_out', type=str, default='')
parser.add_argument('--loo_dataset_for_combs', dest='loo_dataset_for_combs', type=str, default='')
parser.add_argument('--loo_comb_of_all_datasets', action=argparse.BooleanOptionalAction)

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
# device_map = "auto"
# print('+++++ Using device_map auto ++++')

hf_access_token = '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'

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
    total = torch.cuda.get_device_properties(0).total_memory / (2**30)
    reserved = torch.cuda.memory_reserved(0) / (2**30)
    allocated = torch.cuda.memory_allocated(0) / (2**30)
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

def get_past_runs_info(run_name, train_type):
    pass

    if os.path.exists('../Results'):
        result_dir = f'../Results'
    elif os.path.exists('Results'):
        result_dir = f'Results'
    else:
        raise Exception('Results directory not found')

    past_runs_filename = f'{result_dir}/{run_name}_past_runs.csv'
    write_header = False if os.path.isfile(past_runs_filename) else True

    with open(past_runs_filename, 'a') as past_runs_file:
        if train_type == 'LOO':
            writer = csv.DictWriter(past_runs_file,
                                    fieldnames=['loo_dataset', 'comb_dataset', 'comb_num', 'seed',])
        elif train_type == 'PAIR':
            writer = csv.DictWriter(past_runs_file,
                                    fieldnames=['pair_datasets', 'comb_dataset', 'comb_num', 'seed',])

        if write_header:
            writer.writeheader()

    return pd.read_csv(past_runs_filename)

def save_past_runs_info(df_past_runs_info, run_name):
    if os.path.exists('../Results'):
        result_dir = f'../Results'
    elif os.path.exists('Results'):
        result_dir = f'Results'
    else:
        raise Exception('Results directory not found')

    past_runs_filename = f'{result_dir}/{run_name}_past_runs.csv'
    df_past_runs_info.to_csv(past_runs_filename, index=False)

def set_run_params(params, repo_id):
    print(f"Training with hyperparameters: {params}")

    print(f'training {repo_id}')

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=repo_id,
        max_steps=params['max_steps'],
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
    )

    # training_args = update_training_arguments(training_args, **training_args_update)
    lora_args = {'rank': params['lora_rank'], 'alpha': params['lora_alpha']}

    return training_args, lora_args


def train_loo():
    # First, check that param_combination_file is exists
    if not args.param_combination_file:
        raise ValueError('Please provide a file with parameter combinations')

    if not args.leave_out:
        raise ValueError('Please provide a dataset to leave out')

    # Check if using combinations of all datasets
    # If not, load the combination of the desired dataset
    if not args.loo_comb_of_all_datasets:
        if not args.loo_dataset_for_combs:
            raise ValueError('Please provide a dataset to load the combinations from')
        if args.loo_dataset_for_combs not in args.loo_datasets:
            raise ValueError('The dataset for combinations is not in the datasets list')
        if args.loo_dataset_for_combs == args.leave_out:
            raise ValueError('The dataset for combinations cannot be the same as the leave out dataset')

        datasets_for_combs = [args.loo_dataset_for_combs]

    else:
        datasets_for_combs = args.loo_datasets
        if args.leave_out in datasets_for_combs:
            datasets_for_combs.remove(args.leave_out)

    train_loo_name = 'LOO_' + args.leave_out
    combs_format_name = 'COMB_' + args.loo_dataset_for_combs if not args.loo_comb_of_all_datasets else 'ALL'

    df_combs = pd.read_csv(args.param_combination_file)

    data_dict = load_LOO_datasets(datasets=args.loo_datasets, add_intructions=True, with_val=False,
                                  data_percent=args.data_percent)
    data_dict = load_current_LOO(train_names=args.loo_datasets, test_name=args.leave_out,
                                 all_datasets_dict=data_dict, with_val=False)

    df_past_runs_info = get_past_runs_info(train_loo_name, train_type='LOO')

    print('Running Leave One out for dataset: ', args.leave_out)
    for dataset_name in datasets_for_combs:
        print('Training on combinations of dataset: ', dataset_name)

        # Get the combinations for this dataset
        curr_dataset_combs = df_combs[df_combs['dataset'] == dataset_name]

        for comb in curr_dataset_combs.iterrows():
            current_params = comb[1].to_dict()
            # remove model name so that it won't override the current model name
            current_params.pop('model_name', None)

            print(f'** Training comb #{current_params["top"]} **')

            for seed in args.seeds:
                print(f'** Training with seed = {seed} **')

                curr_run_info = {'loo_dataset': args.leave_out,
                                 'comb_dataset': dataset_name,
                                 'comb_num': current_params['top'],
                                 'seed': seed}

                if ((df_past_runs_info[list(curr_run_info)] == pd.Series(curr_run_info)).all(axis=1)).any():
                    print('~~ Combination and split already exists. ~~')
                    print(curr_run_info)
                    print('~~ Skipping to the next combination. ~~')
                    continue

                current_params['seed'] = seed

                # REPOSITORY_ID = f"{prefix}{args.base_model_name.split('/')[1]}-{train_loo_name}-{combs_format_name}-" \
                #                 f"{datetime.now().date()}"

                REPOSITORY_ID = f"{REPO_ID_PREFIX}{args.base_model_name.split('/')[1]}" \
                                 f"-{train_loo_name}-{combs_format_name}-" \
                                f"comb{current_params['top']}-seed{seed}-NEW" \
                                f"{datetime.now().date()}"

                # Set the run parameters
                training_args, lora_args = set_run_params(current_params, REPOSITORY_ID)
                # print(f"Training with hyperparameters: {current_params}")
                #
                # # Define the training arguments
                # training_args = TrainingArguments(
                #     output_dir=REPOSITORY_ID,
                #     max_steps=current_params['max_steps'],
                #     per_device_train_batch_size=current_params['per_device_train_batch_size'],
                #     per_device_eval_batch_size=current_params['per_device_eval_batch_size'],
                #     learning_rate=current_params['learning_rate'],
                #     seed=current_params['seed'],
                #     # hub_model_id=REPOSITORY_ID,
                #     hub_token=HfFolder.get_token(),
                #     gradient_accumulation_steps=4,
                #     logging_steps=10,
                #     report_to='none',
                #     gradient_checkpointing=True,
                # )
                #
                # # training_args = update_training_arguments(training_args, **training_args_update)
                # lora_args = {'rank': current_params['lora_rank'], 'alpha': current_params['lora_alpha']}

                print(f'training {REPOSITORY_ID}')
                _train(output_dir=REPOSITORY_ID, training_args=training_args,
                            data_dict=data_dict, lora_args=lora_args, additional_run_args=current_params,
                            is_cross_val=False, train_dataset_name=train_loo_name,
                            eval_datasets_names=args.loo_datasets)
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()

                curr_run_info = {k: [v] for k, v in curr_run_info.items()}
                df_past_runs_info = pd.concat([df_past_runs_info, pd.DataFrame(curr_run_info)], ignore_index=True)
                save_past_runs_info(df_past_runs_info, train_loo_name)


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

        # TODO Mor: my additions, remove it later
        # backup_model = model
        # from peft import PeftModel
        # peft_model = get_peft_model(model, model.peft_config['default'])
        # trainer = SFTTrainer(model=model, dataset_text_field='instruction', max_seq_length=1024,
                             # peft_config=model.peft_config['default'])

        # print(f'backup_model == trainer.model: {backup_model == trainer.model}')


        for eval_dataset_name in eval_datasets:
            start_eval_time = time.time()
            if is_cross_eval:
                # Load specific split of cross validation of the requested dataset
                dataset, kf = load_cv_dataset(num_of_split=4, dataset_name=eval_dataset_name,
                                              percent=args.data_percent, add_instruction=True,
                                              with_val=False)
                train_indices, test_indices = list(kf.split(dataset['instruction'], dataset['label']))[cv_split_num]
                # train = Dataset.from_pandas(dataset.iloc[train_indices])
                test = Dataset.from_pandas(dataset.iloc[test_indices])
                test = test.class_encode_column("label")
                test = test.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')['test']

                curr_dataset = DatasetDict()
                curr_dataset['train'] = None
                curr_dataset['test'] = test

            else:
                # pay attention I changed the percent to test data percent
                # actually test_data_percent here is used as the validation test.
                # According to this we will choose parameters and model
                # When we evaluate it will be 10%, and when we compute the real model it will be 90%

                curr_dataset = load_dataset(eval_dataset_name, instruction_version=args.instruction_version,
                                            test_percent=args.test_data_percent, add_instruction=True, with_val=False,
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
                if '### Response:\n' not in response:
                    return 'Illegal'

                response_idx = response.index('### Response:\n') + len('### Response:\n')
                response = response[response_idx:]
                if 'Yes' in response and 'No' in response:
                    return 'Illegal'
                if 'Yes' in response:
                    return 'Yes'
                if 'No' in response:
                    return 'No'
                return 'Illegal'

            # TODO Mor: remove it later
            for i, curr_model in enumerate([model]):
                print(f'*** Model {"Backup" if i == 0 else "trainer.model"} ***')

                true_labels = curr_dataset[split].map(get_response)

                # Remove the response from the instruction (If there is one)
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
                    #TODO Mor: remove it later
                    # outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"),
                    #                          attention_mask=inputs["attention_mask"],
                    #                          max_new_tokens=args.max_new_tokens,
                    #                          pad_token_id=tokenizer.eos_token_id)
                    outputs = curr_model.generate(input_ids=inputs["input_ids"].to("cuda"),
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

                run_args = {'train_dataset': train_dataset_name,
                            'evaluate_dataset': eval_dataset_name,
                            'model_name': only_model_name,
                            'save_dir': output_dir}

                run_args.update(additional_run_args)

                create_report(label_list_int, prediction_list_int, run_args, pos_label=1)

            curr_dataset[split] = curr_dataset[split].add_column("prediction", prediction_list)
            curr_dataset[split].to_csv(f'{output_dir}/{eval_dataset_name}_predictions.csv')

            # with open(f'{output_dir}/instruction_results.txt', 'a') as f:
            #     f.write(f'@@@ {eval_dataset_name} results')
            #     for i in range(len(all_responses)):
            #         f.write(
            #             f'*** {i} ***\ntext true response:\n{true_labels[i]}\nmodel output:\n{all_responses[i]}\n\n')

            end_eval_time = time.time()
            print(f'Evaluation on {eval_dataset_name} took {(end_eval_time - start_eval_time) / 60} minutes')

    # Clear memory
    print('|||||', 'Evaluation: Clearing memory', '|||||', sep='\n')
    import gc
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def train_combined_dataset():
    # First, check that param_combination_file is exists
    if not args.param_combination_file:
        raise ValueError('Please provide a file with parameter combinations')

    if not args.combined_datasets:
        raise ValueError('Please provide list of datasets to combine')

    if len(args.combined_datasets) == 1:
        raise ValueError('Please provide at least two datasets to combine')
    elif len(args.combined_datasets) == 2:
        combination_type = 'PAIR'
    elif len(args.combined_datasets) == 3:
        combination_type = 'TRIO'

    train_combined_name = f'{combination_type}_' + '_'.join(args.combined_datasets)

    df_combs = pd.read_csv(args.param_combination_file)

    data_dict = load_combined_dataset(datasets_names=args.combined_datasets, add_instruction=True,
                                      instruction_version=0, percent=args.data_percent)

    df_past_runs_info = get_past_runs_info(train_combined_name, train_type=combination_type)

    print('Running Combined Training for datasets: ', ', '.join(args.combined_datasets))
    for comb_dataset_name in args.combined_datasets:
        print('Training on combinations of dataset: ', comb_dataset_name)

        # Get the combinations for this dataset
        curr_dataset_combs = df_combs[df_combs['dataset'] == comb_dataset_name]

        for comb in curr_dataset_combs.iterrows():
            current_params = comb[1].to_dict()
            # remove model name so that it won't override the current model name
            current_params.pop('model_name', None)

            print(f'** Training comb #{current_params["top"]} **')

            for seed in args.seeds:
                print(f'** Training with seed = {seed} **')

                curr_run_info = {'pair_datasets': train_combined_name,
                                 'comb_dataset': comb_dataset_name,
                                 'comb_num': current_params['top'],
                                 'seed': seed}

                if ((df_past_runs_info[list(curr_run_info)] == pd.Series(curr_run_info)).all(axis=1)).any():
                    print('~~ Combination and split already exists. ~~')
                    print(curr_run_info)
                    print('~~ Skipping to the next combination. ~~')
                    continue

                current_params['seed'] = seed

                REPOSITORY_ID = f"{REPO_ID_PREFIX}{args.base_model_name.split('/')[1]}" \
                                f"-{train_combined_name}-COMB-{comb_dataset_name}-" \
                                f"comb-{current_params['top']}-seed-{seed}-" \
                                f"{datetime.now().date()}"

                # Set the run parameters
                training_args, lora_args = set_run_params(current_params, REPOSITORY_ID)

                print(f'training {REPOSITORY_ID}')
                _train(output_dir=REPOSITORY_ID, training_args=training_args,
                            data_dict=data_dict, lora_args=lora_args, additional_run_args=current_params,
                            is_cross_val=False, train_dataset_name=train_combined_name,
                            eval_datasets_names=args.eval_datasets)
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()

                curr_run_info = {k: [v] for k, v in curr_run_info.items()}
                df_past_runs_info = pd.concat([df_past_runs_info, pd.DataFrame(curr_run_info)], ignore_index=True)
                save_past_runs_info(df_past_runs_info, train_combined_name)

def train_hyperparams():
    # Load cross-validation splits
    # note to self: think of which instruction id to send (None or 0)
    dataset, kf = load_cv_dataset(num_of_split=4, dataset_name=args.train_dataset,
                                  percent=args.data_percent, add_instruction=True,
                                  with_val=False)
    # Define the hyperparameter space
    param_grid = {
        'seed': args.seeds,
        'learning_rate': args.learning_rates,
        'per_device_train_batch_size': args.batch_sizes,
        'per_device_eval_batch_size': args.batch_sizes,
        'max_steps': args.num_steps,
        'lora_rank': args.lora_ranks,
        'lora_alpha': args.lora_alpha,
    }

    # Generate and shuffle (to randomize the selection) all possible combinations of parameters
    all_combinations = list(product(*param_grid.values()))
    random.shuffle(all_combinations)

    n_iter = 45

    df_all_combs = create_combinations_file(args.train_dataset)

    if not df_all_combs.empty:
        # Count the number of combinations used for the first split
        df_combs_split_0 = df_all_combs[df_all_combs['split_num'] == 0]
        selected_combinations, all_combinations = get_existing_combinations(
            df_combs_split_0.drop(columns=['split_num']), all_combinations)

        print(f'**** Found {len(selected_combinations)} combinations from split 0 ****')

        n_combs_to_add = n_iter - len(selected_combinations)
        selected_combinations += all_combinations[:n_combs_to_add]

    else:
        selected_combinations = all_combinations[:n_iter]

    # Loop through the cross-validation splits
    for split_num, split_indices in enumerate(kf.split(dataset['instruction'], dataset['label'])):
        print('Training on split number:', split_num)
        train_split, test_split = split_indices
        train = Dataset.from_pandas(dataset.iloc[train_split])
        test = Dataset.from_pandas(dataset.iloc[test_split])
        data_dict = DatasetDict()
        data_dict['train'] = train
        data_dict['test'] = test

        # Loop through the selected combinations
        for comb_id, params in enumerate(selected_combinations):
            current_params = dict(zip(param_grid.keys(), params))
            print(f'Combination {comb_id + 1}/{n_iter}')
            print(f"Training with hyperparameters: {current_params}")

            params_and_split = current_params.copy()
            params_and_split['split_num'] = split_num
            # Check if the combination and split is already in the file
            if ((df_all_combs[list(params_and_split)] == pd.Series(params_and_split)).all(axis=1)).any():
                print('~~ Combination and split already exists. ~~')
                print(params_and_split)
                print('~~ Skipping to the next combination. ~~')
                continue

            REPOSITORY_ID = f"{REPO_ID_PREFIX}{args.base_model_name.split('/')[1]}-{args.train_dataset}-" \
                            f"{datetime.now().date()}"

            print(f'training {REPOSITORY_ID}')

            # Define the training arguments
            training_args = TrainingArguments(
                output_dir=REPOSITORY_ID,
                max_steps=current_params['max_steps'],
                per_device_train_batch_size=current_params['per_device_train_batch_size'],
                per_device_eval_batch_size=current_params['per_device_eval_batch_size'],
                learning_rate=current_params['learning_rate'],
                seed=current_params['seed'],
                # hub_model_id=REPOSITORY_ID,
                hub_token=HfFolder.get_token(),
                gradient_accumulation_steps=4,
                logging_steps=10,
                report_to='none',
                gradient_checkpointing=True,
            )

            lora_args = {'rank': current_params['lora_rank'], 'alpha': current_params['lora_alpha']}

            from copy import deepcopy
            additional_run_args = deepcopy(current_params)
            additional_run_args['split_num'] = split_num

            _train(output_dir=REPOSITORY_ID, training_args=training_args,
                        data_dict=data_dict, lora_args=lora_args, additional_run_args=additional_run_args,
                        is_cross_val=True, cv_split_num=split_num)


def create_combinations_file(dataset_name):
    PARAM_NAMES = ['seed', 'learning_rate', 'per_device_train_batch_size',
                   'per_device_eval_batch_size', 'max_steps', 'lora_rank', 'lora_alpha', 'split_num']

    if os.path.exists('../Results'):
        results_dir = '../Results'
    elif os.path.exists('Results'):
        results_dir = 'Results'
    else:
        raise Exception('Results directory not found')

    dataset_result_dirname = 'Llama-2-7b-hf-{DATASET}-2024-09-'.format(DATASET=dataset_name)

    df_all_combs = pd.DataFrame(columns=PARAM_NAMES)
    for _, dirs, _ in os.walk(results_dir):

        for dir_name in dirs:
            # Check if the results directory is of the wanted dataset
            if dataset_result_dirname in dir_name:
                inner_dir = os.path.join(results_dir, dir_name)
                score_file_path = os.path.join(inner_dir, f'{dataset_name}_scores.csv')
                df = pd.read_csv(score_file_path)
                df = df[df['evaluate_dataset'] == dataset_name]

                # Get only the parameters columns and append them to the overall parameters dataframe
                df_only_params = df[PARAM_NAMES]
                df_all_combs = pd.concat([df_all_combs, df_only_params])

            else:
                continue

        # Bread after one level of directories, not entering the inner directories
        break
    if not df_all_combs.empty:
        df_all_combs.to_csv(os.path.join(results_dir, f'{dataset_name}_all_combs.csv'), index=False)

    return df_all_combs


def train_loo_with_few():
    if not args.leave_out:
        raise ValueError('Please provide a dataset to leave out')
    if not args.param_combination_file:
        raise ValueError('Please provide a file with parameter combinations')

    train_loo_name = 'LOO_' + args.leave_out
    train_loo_few_name = 'LOO_WITH_FEW_' + args.leave_out

    # Get model name and params of best model of loo dataset
    df_combs = pd.read_csv(args.param_combination_file)
    df_combs = df_combs[df_combs['dataset'] == train_loo_name]
    df_combs = df_combs[df_combs['top'] == 1]

    train_comb_params = df_combs.iloc[0].to_dict()
    # remove model name so that it won't override the current model name
    train_comb_params.pop('model_name', None)

    # load dataset, only 20 samples for the train
    data_dict = load_LOO_datasets(datasets=args.loo_datasets, add_intructions=True, with_val=False,
                                  data_percent=args.data_percent)
    data_dict = load_current_LOO(train_names=args.loo_datasets, test_name=args.leave_out,
                                 all_datasets_dict=data_dict, with_val=False, loo_with_few=True)

    print('Running Leave One Out + Few for dataset: ', args.leave_out)

    for seed in args.seeds:
        print(f'** Training with seed = {seed} **')
        train_comb_params['seed'] = seed

        REPOSITORY_ID = f"{REPO_ID_PREFIX}{args.base_model_name.split('/')[1]}" \
                        f"-{train_loo_few_name}" \
                        f"-seed{seed}" \
                        f"-{datetime.now().date()}"

        # Set the run parameters
        training_args, lora_args = set_run_params(train_comb_params, REPOSITORY_ID)

        print(f'training {REPOSITORY_ID}')
        _train(output_dir=REPOSITORY_ID, training_args=training_args,
                    data_dict=data_dict, lora_args=lora_args, additional_run_args=train_comb_params,
                    is_cross_val=False, train_dataset_name=train_loo_few_name)

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()



def _train(output_dir=None, training_args=None, data_dict=None, lora_args=None,
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
        assert len(args.num_steps) == 1
        assert len(args.lora_ranks) == 1
        assert len(args.lora_alpha) == 1

        print('** Loading training arguments from command line arguments **')
        # Set the run parameters
        current_params = {
            'seed': args.seeds[0],
            'learning_rate': args.learning_rates[0],
            'per_device_train_batch_size': args.batch_sizes[0],
            'per_device_eval_batch_size': args.batch_sizes[0],
            'max_steps': args.num_steps[0],
            'lora_rank': args.lora_ranks[0],
            'lora_alpha': args.lora_alpha[0],
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
        dataset = load_dataset(args.train_dataset, instruction_version=args.instruction_version,
                               percent=args.data_percent, add_instruction=True, with_val=False)
        train_dataset = dataset['train']

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
    if args.task == 'TRAIN':
        _train()

    if args.task == 'EVAL':
        evaluate()

    if args.task == 'TRAIN_HYPERPARAMS':
        train_hyperparams()

    if args.task == 'LOO':
        train_loo()

    if args.task == 'LOO_WITH_FEW':
        train_loo_with_few()

    if args.task == 'COMBINED':
        train_combined_dataset()

    if args.task == '':
        print('Nothing were given to do. Please provide a task to do.')
