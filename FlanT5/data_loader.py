import os

import pandas
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import KFold, StratifiedKFold

from typing import Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATASET_NAME = 'yelp'
DATASET_FIXED_SIZE = 19000
INSTRUCTION = 'Read the following text and classify it as funny or not funny. Generate only ONE word, Yes for funny, No' \
              'for not funny\n\n'

label2id = {"not funny": 0, "funny": 1}
humorous2id = {"not humorous": 0, "humorous": 1}
id2label = {id: label for label, id in label2id.items()}
id2humorous = {id: label for label, id in humorous2id.items()}


def create_instruction(version_idx):
    INSTRUCTION_VERSIONS = [
        "Given the following text, please determine if it should be classified as funny or not funny."
        " Base your classification on humor elements such as wit, irony, absurdity, or comedic timing.",

        "Analyze the text below and decide if it is funny or not funny. Consider factors like wordplay,"
        " jokes, puns, sarcasm, or any other comedic techniques."
        " Clearly label the text as 'Funny' or 'Not Funny'.",

        "Does the following text exhibit characteristics of humor, such as being witty, sarcastic, or absurd?"
        " Respond with 'Funny' for yes and 'Not Funny' for no.",

        "Read the text provided and assess its humor potential. Would it likely make someone laugh or smile?"
        " Classify the text as either 'Funny' or 'Not Funny'.",

        "Imagine someone telling this text as a joke or an amusing story."
        " In this context, would you find it funny? Categorize the text as 'Funny' or 'Not Funny'.",

        "Consider the emotional response the text is likely to provoke."
        " If it tends to make people laugh or find it amusing, label it as 'Funny'."
        " If it does not, label it as 'Not Funny'."
    ]

    if version_idx is not None:
        version_text = INSTRUCTION_VERSIONS[version_idx]

        text_input_format = "Below is an instruction that describes a sentiment analysis task.\n\n" \
                            "### Instruction:\n" + version_text + "\n\n### Input:\n{text}\n\n### Response:\n"

    else:
        text_input_format = "Below is an instruction that describes a text classification task.\n\n" \
                            "### Instruction:\nAnalyze the following text and classify whether it is funny or not." \
                            " If it is funny output Yes, and if it isn't funny output No.\n\n" \
                            "### Input:\n{text}\n\n" \
                            "### Response:\n"

    def apply_to_row(row):
        return text_input_format.format(text=row['text'])  # , label=id2humorous[row['label']])
        # return text_to_model_format.format(text=row['text'])

    return apply_to_row


def add_response(row):
    output = 'Yes' if row['label'] == 1 else 'No'
    row['instruction'] = row['instruction'] + output
    # row['instruction'] = row['instruction'] + id2humorous[row['label']]
    return row


def load_dataset(dataset_name='amazon', percent=None, data_file_path=None,
                 add_instruction: bool = False, instruction_version=0,
                 with_val=True, train_percent=None) -> DatasetDict:
    """Load dataset."""
    if not data_file_path:
        data_file_path = ROOT_DIR + f"/Data/new_humor_datasets/balanced/{dataset_name}/data.csv"
    else:
        print('***', 'Used data file path:', data_file_path, '***', sep='\n')

    dataset_pandas = pd.read_csv(data_file_path)

    # if percent and type(percent) is float and percent <= 100:
    #     samples_count = int(len(dataset_pandas) * percent / 100)
    #     dataset_pandas = dataset_pandas.iloc[:samples_count]

    dataset_pandas["text"] = dataset_pandas["text"].astype(str)

    if add_instruction:
        dataset_pandas["instruction"] = dataset_pandas.apply(create_instruction(instruction_version), axis=1)

    dataset = Dataset.from_pandas(dataset_pandas)
    dataset = dataset.class_encode_column("label")

    if percent and type(percent) is float and percent <= 1:
        dataset = dataset.train_test_split(test_size=percent, seed=42, stratify_by_column='label')['test']

    if with_val:
        # 75% train, 25% test + validation
        train_testval = dataset.train_test_split(test_size=0.25, seed=42, stratify_by_column='label')
        train = train_testval['train']
        # Split the 25% test + valid in 40% test, 60% valid
        test_valid = train_testval['test'].train_test_split(test_size=0.4, seed=42, stratify_by_column='label')
        test, val = test_valid['test'], test_valid['train']

    else:
        # 80% train, 20% test
        train_test = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='label')
        train = train_test['train']
        test = train_test['test']
        val = None

        if train_percent and type(train_percent) is float and train_percent <= 1:
            train = train.train_test_split(test_size=train_percent, seed=42, stratify_by_column='label')['test']

    if add_instruction:
        train = train.map(add_response)

    # gather everyone if you want to have a single DatasetDict
    dataset_dict = DatasetDict({
        'train': train,
        'test': test,
        'val': val})

    return dataset_dict


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


def load_LOO_datasets(datasets, add_intructions=False, instruction_version=0,
                      with_val=True):
    """ load leave one out datasets"""
    data_path = ROOT_DIR + "/Data/new_humor_datasets/balanced/{dataset_name}/data.csv"
    all_datasets_dict = {}

    TRAIN_DATA_PERCENT = 1 / (len(datasets) - 1)

    for dataset in datasets:
        all_datasets_dict[dataset] = load_dataset(dataset, train_percent=TRAIN_DATA_PERCENT,
                                                  add_instruction=add_intructions,
                                                  instruction_version=instruction_version,
                                                  with_val=with_val)
        # df = pd.read_csv(data_path.format(dataset_name=dataset))
        #
        # # append eos token to the other datasets to align them with amount of eos of amazon
        # # if dataset != 'amazon':
        # #     df['text'] = df['text'].apply(lambda s: s + ' </s>')
        #
        # # df = df.iloc[:1000]
        #
        # dataset_df = Dataset.from_pandas(df)
        # dataset_df = dataset_df.class_encode_column("label")
        # # 10% test, 90% train + validation
        # trainval_test = dataset_df.train_test_split(test_size=0.2, seed=42, stratify_by_column='label')
        #
        # # Split the 90% train + valid in 85% train, 15% valid
        # train_valid = trainval_test['train'].train_test_split(test_size=0.15, seed=42, stratify_by_column='label')
        # test = trainval_test['test']
        #
        # # Divide each of train, valid by number of dataset for training (len(datasets)-1)
        # todrop_train = train_valid['train'].train_test_split(test_size=DATA_PERCENT, seed=42,
        #                                                      stratify_by_column='label')
        # train = todrop_train['test']
        # todrop_val = train_valid['test'].train_test_split(test_size=DATA_PERCENT, seed=42, stratify_by_column='label')
        # val = todrop_val['test']
        #
        # dataset_dict = DatasetDict({
        #     'train': train,
        #     'test': test,
        #     'val': val})

        # all_datasets_dict[dataset] = dataset_dict

    return all_datasets_dict


def load_cv_dataset(num_of_split=5, dataset_name='amazon', percent=None,
                    data_file_path=None, add_instruction: bool = False,
                    instruction_version=0, with_val=True
                    ) -> Tuple[pandas.DataFrame, StratifiedKFold]:
    # dataset_pandas = pd.read_csv(
    #     ROOT_DIR + f"/Data/new_humor_datasets/balanced/{dataset_name}/data.csv"
    # )
    #
    # if percent and type(percent) is float and percent <= 100:
    #     samples_count = int(len(dataset_pandas) * percent / 100)
    #     dataset_pandas = dataset_pandas.iloc[:samples_count]
    #
    # dataset_pandas["text"] = dataset_pandas["text"].astype(str)
    # dataset = Dataset.from_pandas(dataset_pandas)
    # # 90% train, 10% test
    # dataset = dataset.class_encode_column("label")
    # train_test = dataset.train_test_split(test_size=0.25, seed=42, stratify_by_column='label')

    dataset_dict = load_dataset(dataset_name, percent, data_file_path,
                                add_instruction, with_val, instruction_version)

    kf = StratifiedKFold(n_splits=num_of_split, shuffle=True, random_state=1)

    train = pd.DataFrame(dataset_dict['train'])

    return train, kf


from datasets import Dataset
from sklearn.model_selection import train_test_split


def stratified_sample(dataset, label_column, sample_fraction=0.1):
    # Extract labels
    labels = dataset[label_column]

    # Perform stratified sampling using train_test_split
    _, sampled_indices = train_test_split(
        range(len(dataset)),
        test_size=sample_fraction,
        stratify=labels,
        random_state=42  # Ensure reproducibility
    )

    # Select the sampled subset
    sampled_dataset = dataset.select(sampled_indices)

    return sampled_dataset


if __name__ == "__main__":
    dataset, kf = load_cv_dataset(dataset_name='headlines', add_instruction=False, with_val=False,
                                  percent=1)

    for split in ['train', 'test']:
        print(f'split = {split}')
        for sample in dataset[split]:
            print(sample['instruction'])
            print(sample['label'])

    # load_dataset(add_instruction=False)
    pass
