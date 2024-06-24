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
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import csv
from data_loader import id2label, load_dataset, load_cv_dataset
from datasets import DatasetDict
import sys

sys.path.append('../')
from Utils.utils import print_cur_time

# Load the model and tokenizer
MODEL_ID = "morturr/flan-t5-base-amazon-text-classification"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.to("cuda") if torch.cuda.is_available() else model.to("cpu")

DATASET_NAME = 'amazon'
datasets_list = ['amazon', 'yelp', 'sarcasm_headlines']
dataset = load_dataset("AutoModelForSequenceClassification", DATASET_NAME)
# dataset, kf = load_cv_dataset("AutoModelForSequenceClassification")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


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

    # Map predicted class IDs to labels
    predicted_labels = [id2label[class_id] for class_id in predicted_classes]

    # Zip together the predicted labels and confidences and convert to a list of tuples
    return list(zip(predicted_labels, confidences))


def evaluate():
    """Evaluate the model on the test dataset."""
    for dataset_name in datasets_list:
        dataset = load_dataset("AutoModelForSequenceClassification", dataset_name)

        print_cur_time(f'***** Evaluate model: {MODEL_ID} on dataset: {dataset_name} *****')
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
        report = classification_report(labels_list, [pair[0] for pair in predictions_list])
        print(report)


def evaluate_with_cv(data_dict, model_name, run_args):
    global model

    model = AutoModelForSequenceClassification.from_pretrained(f'morturr/{model_name}')
    model.to("cuda") if torch.cuda.is_available() else model.to("cpu")

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

    # report = classification_report(labels_list, [pair[0] for pair in predictions_list])
    report = classification_report(labels_list, predictions_list)
    accuracy = accuracy_score(labels_list, predictions_list)
    precision = precision_score(labels_list, predictions_list, pos_label='funny')
    recall = recall_score(labels_list, predictions_list, pos_label='funny')
    f1 = f1_score(labels_list, predictions_list, pos_label='funny')

    result_filename = f'{run_args["dataset_name"]}_scores.csv'
    write_header = False if os.path.isfile(result_filename) else True

    with open(result_filename, 'a') as csvfile:
        results_dict = {
            'model': model_name,
            'dataset': run_args['dataset_name'],
            'epoch': run_args['epoch'],
            'batch_size': run_args['batch_size'],
            'learning_rate': run_args['learning_rate'],
            'seed': run_args['seed'],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        writer = csv.DictWriter(csvfile,
                                fieldnames=['model', 'dataset',
                                            'epoch', 'batch_size', 'learning_rate',
                                            'seed', 'accuracy', 'precision',
                                            'recall', 'f1'])
        if write_header:
            writer.writeheader()
        writer.writerow(results_dict)

    print(report)
    print(results_dict)
    print('*******************************')

    with open(f'{run_args["dataset_name"]}_reports.txt', 'a') as report_file:
        report_file.write(f'model name: {model_name}\n')
        report_file.write(f'dataset name: {run_args["dataset_name"]}\n')
        report_file.write(report)
        report_file.write('\n*************\n')


if __name__ == "__main__":
    evaluate()
