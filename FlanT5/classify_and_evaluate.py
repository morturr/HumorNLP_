from time import time
from typing import List, Tuple

import torch
from loguru import logger
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)

import os.path
import os
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
import csv
from FlanT5.data_loader import id2label, load_dataset
# from flan_utils import FlanEvaluationArguments
from FlanT5.flan_utils import FlanTrainingArguments, FlanEvaluationArguments

from datasets import DatasetDict
import sys

sys.path.append('../')
from Utils.utils import print_cur_time

DATASETS = ['amazon', 'yelp_reviews', 'dadjokes', 'one_liners', 'headlines']


def classify(texts_to_classify: List[str]) -> List[Tuple[str, float]]:
    """Classify a list of texts using the model."""

    # Tokenize all texts in the batch
    start = time()
    inputs = tokenizer(
        texts_to_classify,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    logger.debug(
        f"Classification of {len(texts_to_classify)} examples took {time() - start} seconds"
    )

    # Process the outputs to get the probability distribution
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get the top class and the corresponding probability (certainty) for each input text
    confidences, predicted_classes = torch.max(probs, dim=1)
    predicted_classes = (
        predicted_classes.cpu().numpy()
    )  # Move to CPU for numpy conversion if needed
    confidences = confidences.cpu().numpy()  # Same here

    # # TODO: remove this later, only to check flan logits
    # print('*** predicted classes ***')
    # print(f'type = {type(predicted_classes)}\n value = {predicted_classes}')
    # Map predicted class IDs to labels
    predicted_labels = [id2label[class_id] for class_id in predicted_classes]

    # Zip together the predicted labels and confidences and convert to a list of tuples
    return list(zip(predicted_labels, confidences))


def evaluate():
    """Evaluate the model on the test dataset."""
    if data_args.test_file_path and len(data_args.datasets) > 1:
        raise Exception("Evaluate got test file path but more than a single dataset")

    model_name = MODEL_ID[MODEL_ID.index('morturr/') + len('morturr/'):]

    # TODO: Heuristic to find the trained dataset in model name
    # TODO: PAY ATTENTION TO IT!
    trained_dataset = [name for name in DATASETS if name in model_name][0]
    model_seed = int(MODEL_ID[MODEL_ID.index('seed-') + len('seed-'):])

    run_args = {
        'train_dataset': trained_dataset,
        'model_name': model_name,
        'evaluate_dataset': '',
        'seed': model_seed}

    for eval_dataset_name in data_args.datasets:
        run_args['evaluate_dataset'] = eval_dataset_name

        dataset = load_dataset(dataset_name=eval_dataset_name,
                               percent=data_args.eval_samples_percent,
                               data_file_path=data_args.test_file_path,
                               add_instruction=data_args.add_instruction_test)

        print_cur_time(f'***** Evaluate model: {MODEL_ID} on dataset: {eval_dataset_name} *****')
        predictions_list, labels_list = [], []

        batch_size = 16  # Adjust batch size based GPU capacity
        num_batches = len(dataset["test"]) // batch_size + (
            0 if len(dataset["test"]) % batch_size == 0 else 1
        )
        progress_bar = tqdm(total=num_batches, desc="Evaluating")

        for i in range(0, len(dataset["test"]), batch_size):
            batch_texts = dataset["test"]["text"][i: i + batch_size]
            batch_labels = dataset["test"]["label"][i: i + batch_size]

            batch_predictions = classify(batch_texts)

            predictions_list.extend(batch_predictions)
            labels_list.extend([id2label[label_id] for label_id in batch_labels])

            progress_bar.update(1)

        progress_bar.close()

        predictions_list = [pair[0] for pair in predictions_list]

        if data_args.create_report_files:
            create_report(labels_list, predictions_list, run_args, pos_label='funny')
        else:
            report = classification_report(labels_list, predictions_list)
            print(report)


def evaluate_with_cv(data_dict, run_args):
    global model, tokenizer

    model = AutoModelForSequenceClassification.from_pretrained(f'morturr/{run_args["model_name"]}')
    model.to("cuda") if torch.cuda.is_available() else model.to("cpu")

    tokenizer = AutoTokenizer.from_pretrained(f'morturr/{run_args["model_name"]}')

    """Evaluate the model on the test dataset."""
    predictions_list, labels_list = [], []

    batch_size = 16  # Adjust batch size based GPU capacity
    num_batches = len(data_dict["test"]) // batch_size + (
        0 if len(data_dict["test"]) % batch_size == 0 else 1
    )
    progress_bar = tqdm(total=num_batches, desc="Evaluating")

    for i in range(0, len(data_dict["test"]), batch_size):
        batch_texts = data_dict["test"]["text"][i: i + batch_size]
        batch_labels = data_dict["test"]["label"][i: i + batch_size]

        batch_predictions = classify(batch_texts)

        predictions_list.extend(batch_predictions)
        labels_list.extend([id2label[label_id] for label_id in batch_labels])

        progress_bar.update(1)

    progress_bar.close()
    predictions_list = [pair[0] for pair in predictions_list]

    if data_args.create_report_files:
        create_report(labels_list, predictions_list, run_args, pos_label='funny')
    else:
        report = classification_report(labels_list, predictions_list)
        print(report)


def create_report(labels_list, predictions_list, run_args, pos_label=1):
    report = classification_report(labels_list, predictions_list)
    accuracy = accuracy_score(labels_list, predictions_list)
    precision = precision_score(labels_list, predictions_list, pos_label=pos_label)
    recall = recall_score(labels_list, predictions_list, pos_label=pos_label)
    f1 = f1_score(labels_list, predictions_list, pos_label=pos_label)

    if 'save_dir' in run_args:
        result_path = run_args['save_dir']
    else:
        result_path = f'Results/{run_args["model_name"]}'
    os.makedirs(result_path, exist_ok=True)

    result_filename = f'{result_path}/{run_args["train_dataset"]}_scores.csv'
    write_header = False if os.path.isfile(result_filename) else True

    with open(result_filename, 'a') as csvfile:
        metric_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1}

        merged_results_dict = {**run_args, **metric_dict}
        # results_dict = {
        #     'model': run_args["model_name"],
        #     'train_dataset': run_args['train_dataset'],
        #     'evaluate_dataset': run_args['evaluate_dataset'],
        #     'epoch': run_args['epoch'],
        #     'batch_size': run_args['batch_size'],
        #     'learning_rate': run_args['learning_rate'],
        #     'seed': run_args['seed'],
        #     'accuracy': accuracy,
        #     'precision': precision,
        #     'recall': recall,
        #     'f1': f1
        # }

        writer = csv.DictWriter(csvfile, fieldnames=list(merged_results_dict.keys()))
        # fieldnames=['model', 'train_dataset', 'evaluate_dataset',
        #             'epoch', 'batch_size', 'learning_rate',
        #             'seed', 'accuracy', 'precision',
        #             'recall', 'f1'])
        if write_header:
            writer.writeheader()
        writer.writerow(merged_results_dict)

    print(report)
    print(merged_results_dict)
    print('*******************************')

    with open(f'{result_path}/{run_args["train_dataset"]}_reports.txt', 'a') as report_file:
        report_file.write(f'model name: {run_args["model_name"]}\n')
        if run_args["train_dataset"]:
            report_file.write(f'train dataset: {run_args["train_dataset"]}\n')
        if run_args["evaluate_dataset"]:
            report_file.write(f'evaluate dataset: {run_args["evaluate_dataset"]}\n')
        report_file.write(report)
        report_file.write('\n*************\n')


if __name__ == "__main__":
    # Load the model and tokenizer
    parser = HfArgumentParser((FlanTrainingArguments, FlanEvaluationArguments))
    data_args = parser.parse_args_into_dataclasses()[1]

    for model_id in data_args.models_id:
        # MODEL_ID = "morturr/flan-t5-base-amazon-text-classification"
        MODEL_ID = model_id

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID,
                                                                   cache_dir="/cs/labs/dshahaf/mortur/HumorNLP_/FlanT5/")
        model.to("cuda") if torch.cuda.is_available() else model.to("cpu")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir="/cs/labs/dshahaf/mortur/HumorNLP_/FlanT5/")
        evaluate()
