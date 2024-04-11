import os

import pandas as pd
from datasets import Dataset, DatasetDict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#.replace('\\', '/')

# label2id = {"Books": 0, "Clothing & Accessories": 1, "Electronics": 2, "Household": 3}
label2id = {"not funny": 0, "funny": 1}
id2label = {id: label for label, id in label2id.items()}


def load_dataset(model_type: str = "AutoModelForSequenceClassification") -> DatasetDict:
    """Load dataset."""
    dataset_amazon_pandas = pd.read_csv(
        # ROOT_DIR + "/data/ecommerce_kaggle_dataset.csv",
        ROOT_DIR + "/Data/new_humor_datasets/temp_run/amazon/data.csv",
        # header=None,
        # names=["id", "bert_sentence", "t5_sentence", "target", "label"],
        # names=["label", "text"],
    )
    dataset_amazon_pandas = dataset_amazon_pandas.iloc[:1000]

    # dataset_amazon_pandas["label"] = dataset_amazon_pandas["label"].astype(str)
    # if model_type == "AutoModelForSequenceClassification":
    #     # Convert labels to integers
    #     dataset_amazon_pandas["label"] = dataset_amazon_pandas["label"].map(
    #         label2id
    #     )

    dataset_amazon_pandas["t5_sentence"] = dataset_amazon_pandas["t5_sentence"].astype(str)
    dataset = Dataset.from_pandas(dataset_amazon_pandas)
    dataset = dataset.shuffle(seed=42)
    # 70% train, 30% test + validation
    train_testval = dataset.train_test_split(test_size=0.3)
    # Split the 10% test + valid in half test, half valid
    test_valid = train_testval['test'].train_test_split(test_size=0.5)
    # gather everyone if you want to have a single DatasetDict
    dataset = DatasetDict({
        'train': train_testval['train'],
        'test': test_valid['test'],
        'val': test_valid['train']})

    # dataset = dataset.train_test_split(test_size=0.2)

    return dataset


if __name__ == "__main__":
    print(load_dataset())