import os

import pandas
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import KFold, StratifiedKFold

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATASET_NAME = 'yelp'
DATASET_FIXED_SIZE = 19000

label2id = {"not funny": 0, "funny": 1}
id2label = {id: label for label, id in label2id.items()}


def load_dataset(dataset_name='amazon', percent=None, data_file_path=None) -> DatasetDict:
    """Load dataset."""
    if not data_file_path:
        data_file_path = ROOT_DIR + f"/Data/new_humor_datasets/balanced/{dataset_name}/data.csv"
    else:
        print('***', 'Used data file path:', data_file_path, '***', sep='\n')

    dataset_pandas = pd.read_csv(data_file_path)

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


def get_partial_dataset(dataset_df, divide_by):
    positive = dataset_df[dataset_df['label'] == 1].iloc[:int(size / 2)]
    negative = dataset_df[dataset_df['label'] == 0].iloc[:int(size / 2)]

    new_df = pd.concat([positive, negative], ignore_index=True)
    new_df = new_df.sample(frac=1, random_state=42, ignore_index=True)

    return new_df


def load_current_LOO(train_names, test_name, all_datasets_dict):
    combined_train = Dataset.from_dict({'text': [], 'label': []})
    combined_val = Dataset.from_dict({'text': [], 'label': []})

    for dataset_name in train_names:
        if dataset_name == test_name:
            continue
        combined_train = concatenate_datasets([combined_train, all_datasets_dict[dataset_name]['train']])
        combined_val = concatenate_datasets([combined_val, all_datasets_dict[dataset_name]['val']])

    combined_train = combined_train.shuffle(seed=42)
    combined_val = combined_val.shuffle(seed=42)
    test = all_datasets_dict[test_name]['test']

    dataset_dict = DatasetDict({
        'train': combined_train,
        'test': test,
        'val': combined_val})

    return dataset_dict


def load_LOO_datasets(datasets):
    """ load leave one out datasets"""
    data_path = ROOT_DIR + "/Data/new_humor_datasets/balanced/{dataset_name}/data.csv"
    all_datasets_dict = {}

    DATA_PERCENT = 1 / (len(datasets) - 1)

    for dataset in datasets:
        df = pd.read_csv(data_path.format(dataset_name=dataset))

        # append eos token to the other datasets to align them with amount of eos of amazon
        if dataset != 'amazon':
            df['text'] = df['text'].apply(lambda s: s + ' </s>')

        # df = df.iloc[:1000]

        dataset_df = Dataset.from_pandas(df)
        dataset_df = dataset_df.class_encode_column("label")
        # 10% test, 90% train + validation
        trainval_test = dataset_df.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')

        # Split the 90% train + valid in 85% train, 15% valid
        train_valid = trainval_test['train'].train_test_split(test_size=0.15, seed=42, stratify_by_column='label')
        test = trainval_test['test']

        # Divide each of train, valid by number of dataset for training (len(datasets)-1)
        todrop_train = train_valid['train'].train_test_split(test_size=DATA_PERCENT, seed=42, stratify_by_column='label')
        train = todrop_train['test']
        todrop_val = train_valid['test'].train_test_split(test_size=DATA_PERCENT, seed=42, stratify_by_column='label')
        val = todrop_val['test']

        dataset_dict = DatasetDict({
            'train': train,
            'test': test,
            'val': val})

        all_datasets_dict[dataset] = dataset_dict

    return all_datasets_dict


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


def load_instruction_dataset(model_type: str = "AutoModelForSequenceClassification",
                             dataset_name='amazon') -> DatasetDict:
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
    load_LOO_datasets(['amazon', 'headlines', 'yelp_reviews'], 'yelp_reviews')
    pass
