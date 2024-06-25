import os

import pandas
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import KFold, StratifiedKFold

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATASET_NAME = 'yelp'
DATASET_FIXED_SIZE = 19000

label2id = {"not funny": 0, "funny": 1}
id2label = {id: label for label, id in label2id.items()}


def load_dataset(dataset_name='amazon', percent=None) -> DatasetDict:
    """Load dataset."""
    dataset_pandas = pd.read_csv(
        ROOT_DIR + f"/Data/new_humor_datasets/balanced/{dataset_name}/data.csv",
    )

    if percent and type(percent) is int and percent <= 100:
        samples_count = int(len(dataset_pandas) * percent / 100)
        dataset_pandas = dataset_pandas.iloc[:samples_count]

    dataset_pandas["text"] = dataset_pandas["text"].astype(str)
    dataset = Dataset.from_pandas(dataset_pandas)
    # 70% train, 30% test + validation
    dataset = dataset.class_encode_column("label")
    train_testval = dataset.train_test_split(test_size=0.25, seed=42, stratify_by_column='label')
    # Split the 30% test + valid in half test, half valid
    test_valid = train_testval['test'].train_test_split(test_size=0.4, seed=42, stratify_by_column='label')
    # gather everyone if you want to have a single DatasetDict
    dataset = DatasetDict({
        'train': train_testval['train'],
        'test': test_valid['test'],
        'val': test_valid['train']})

    # dataset = dataset.train_test_split(test_size=0.2)

    return dataset


def load_LOO_dataset(datasets, test_name):
    """ load leave one out datasets"""
    data_path = ROOT_DIR + "/Data/new_humor_datasets/balanced/{dataset_name}/data.csv"
    df_train = pd.DataFrame(columns=['text', 'label'])
    df_test = pd.DataFrame(columns=['text', 'label'])

    for dataset in datasets:
        if dataset == test_name:
            continue
        df = pd.read_csv(data_path.format(dataset_name=dataset))
        df_train = pd.concat([df_train, df[['text', 'label']]], ignore_index=True)


    df = pd.read_csv(data_path.format(dataset_name=test_name))
    df_test = pd.concat([df_test, df[['text', 'label']]], ignore_index=True)

    df_train["text"] = df_train["text"].astype(str)
    df_test["text"] = df_test["text"].astype(str)

    dataset_train = Dataset.from_pandas(df_train)
    dataset_test = Dataset.from_pandas(df_test)

    dataset_train = dataset_train.class_encode_column("label")
    train_val = dataset_train.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')

    # gather everyone if you want to have a single DatasetDict
    dataset = DatasetDict({
        'train': train_val['train'],
        'test': dataset_test,
        'val': train_val['test']})

    return dataset


def load_cv_dataset(num_of_split=5, dataset_name='amazon', percent=None) -> pandas.DataFrame:
    dataset_pandas = pd.read_csv(
        ROOT_DIR + f"/Data/new_humor_datasets/balanced/{dataset_name}/data.csv"
    )

    if percent and type(percent) is int and percent <= 100:
        samples_count = int(len(dataset_pandas) * percent / 100)
        dataset_pandas = dataset_pandas.iloc[:samples_count]

    dataset_pandas["text"] = dataset_pandas["text"].astype(str)
    dataset = Dataset.from_pandas(dataset_pandas)
    # 90% train, 10% test
    dataset = dataset.class_encode_column("label")
    train_test = dataset.train_test_split(test_size=0.25, seed=42, stratify_by_column='label')
    kf = StratifiedKFold(n_splits=num_of_split, shuffle=True, random_state=1)

    train = pd.DataFrame(train_test['train'])

    return train, kf


def load_instruction_dataset(model_type: str = "AutoModelForSequenceClassification", dataset_name='amazon') -> DatasetDict:
    """Load dataset for instruction fine tuning."""
    dataset_pandas = pd.read_csv(
        ROOT_DIR + f"/Data/new_humor_datasets/temp_run/{dataset_name}/data.csv",
        # header=None,
        # names=["id", "bert_sentence", "t5_sentence", "target", "label"],
        # names=["label", "text"],
    )
    dataset_pandas = dataset_pandas.iloc[:10]

    dataset_pandas["t5_sentence"] = dataset_pandas["t5_sentence"].astype(str)
    dataset_pandas["t5_sentence"] = 'Determine if the following text is funny.\nTEXT: ' + \
                                    dataset_pandas["t5_sentence"].astype(str) + \
                                    '\nOPTIONS:\n-funny\n-not funny'

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


if __name__ == "__main__":
    load_LOO_dataset(['amazon', 'headlines', 'yelp_reviews'], 'yelp_reviews')
    pass
