import os

import pandas
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import KFold

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_NAME = 'amazon'
DATASET_FIXED_SIZE = 19000

label2id = {"not funny": 0, "funny": 1}
id2label = {id: label for label, id in label2id.items()}


def load_dataset(model_type: str = "AutoModelForSequenceClassification") -> DatasetDict:
    """Load dataset."""
    dataset_pandas = pd.read_csv(
        ROOT_DIR + f"/Data/new_humor_datasets/temp_run/{DATASET_NAME}/data.csv",
        # header=None,
        # names=["id", "bert_sentence", "t5_sentence", "target", "label"],
        # names=["label", "text"],
    )
    # dataset_pandas = dataset_pandas.iloc[:1000]

    dataset_pandas["t5_sentence"] = dataset_pandas["t5_sentence"].astype(str)
    dataset = Dataset.from_pandas(dataset_pandas)
    # dataset = dataset.shuffle(seed=42)
    # 70% train, 30% test + validation
    train_testval = dataset.train_test_split(test_size=0.3)
    # Split the 30% test + valid in half test, half valid
    test_valid = train_testval['test'].train_test_split(test_size=0.5)
    # gather everyone if you want to have a single DatasetDict
    dataset = DatasetDict({
        'train': train_testval['train'],
        'test': test_valid['test'],
        'val': test_valid['train']})

    # dataset = dataset.train_test_split(test_size=0.2)

    return dataset


def load_cv_dataset(model_type: str = "AutoModelForSequenceClassification", num_of_split=5) -> pandas.DataFrame:
    dataset_pandas = pd.read_csv(
        ROOT_DIR + f"/Data/new_humor_datasets/temp_run/{DATASET_NAME}/data.csv"
    )

    dataset_pandas["t5_sentence"] = dataset_pandas["t5_sentence"].astype(str)
    dataset = Dataset.from_pandas(dataset_pandas)
    # 90% train, 10% test
    train_test = dataset.train_test_split(test_size=0.9, seed=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    return dataset['train'], kf


if __name__ == "__main__":
    print(load_dataset())