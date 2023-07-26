import os
import pandas as pd
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split


class DataPreprocessing:
    def __init__(self):
        self.datasets = {}

    def load_data(self, path, dataset_name):
        train_path = f'{path}/train.csv'
        test_path = f'{path}/test.csv'

        if os.path.exists(train_path) and os.path.exists(test_path):
            train, test = pd.read_csv(train_path), pd.read_csv(test_path)
        else:
            raise FileExistsError('train or test file doesnt exists')

        # train = train.sample(frac=1).reset_index(drop=True)
        ds = DatasetDict()
        ds['train'] = Dataset.from_pandas(train)
        ds['test'] = Dataset.from_pandas(test)

        self.datasets[dataset_name] = ds

    def get_datasets(self):
        return self.datasets

    @staticmethod
    def balance_train(path, datasets):
        for dataset in datasets:
            curr_path = path + dataset + '/'
            train = pd.read_csv(curr_path + 'train.csv')
            test = pd.read_csv(curr_path + 'test.csv')

            train_label_1 = train[train['label'] == 1]
            train_label_0 = train[train['label'] == 0]
            larger_df = train_label_1 if len(train_label_1) > len(train_label_0) else train_label_0
            shorter_df = train_label_0 if len(train_label_1) > len(train_label_0) else train_label_1
            labels_diff = abs(len(train_label_1) - len(train_label_0))

            if labels_diff > 0:
                os.makedirs(curr_path + 'before_balance/', exist_ok=True)
                test.to_csv(curr_path + 'before_balance/' + 'test.csv')
                train.to_csv(curr_path + 'before_balance/' + 'train.csv')
                test = test.append(larger_df.iloc[:labels_diff], ignore_index=True)
                larger_df = larger_df.iloc[labels_diff:]
                test = test.sample(frac=1, random_state=0, ignore_index=True)
                train = larger_df.append(shorter_df)
                train = train.sample(frac=1, random_state=0, ignore_index=True)
                test.to_csv(curr_path + 'test.csv')
                train.to_csv(curr_path + 'train.csv')

    @staticmethod
    def load_amazon():
        path = './original_datasets/amazon/all_data.csv'
        df = pd.read_csv(path)
        df_new = pd.DataFrame()
        df_new['sentence'] = df['question'] + ' [SEP] ' + df['product_description']
        df_new['label'] = df['label']
        df_new = df_new[df_new['sentence'].notna()]
        df_new = df_new.sample(frac=1, random_state=0)
        df_new['idx'] = range(0, len(df_new))
        df_new.to_csv('./humor_datasets/amazon/data.csv', index=False)

        train, test = train_test_split(df_new, test_size=0.2, shuffle=True)
        train.to_csv('./humor_datasets/amazon/train.csv', index=False)
        test.to_csv('./humor_datasets/amazon/test.csv', index=False)

    @staticmethod
    def load_headlines():
        def edit_headline(row):
            headline = row['original']
            edit_word = row['edit']
            res = headline[:headline.index('<')] + edit_word + headline[headline.index('>') + 1:]
            return res

        splits = ['train', 'dev', 'test', 'train_funlines']

        for split in splits:
            path = f'./original_datasets/headlines/{split}.csv'
            df = pd.read_csv(path)
            df = df[df['edit'].notna()]
            df_new = pd.DataFrame()
            df_new['idx'] = df['id']
            df_new['meanGrade'] = df['meanGrade']
            df_new['sentence'] = df.apply(edit_headline, axis=1)
            df_new['label'] = df.apply(lambda row: 1 if row['meanGrade'] >= 1 else 0, axis=1)
            df_new.to_csv(f'./humor_datasets/headlines/{split}.csv', index=False)

    @staticmethod
    def load_twss():
        path = './original_datasets/twss/all_data.csv'
        df_new = pd.read_csv(path)
        df_new = df_new[df_new['sentence'].notna()]
        df_new = df_new.sample(frac=1, random_state=0, ignore_index=True)
        df_new['idx'] = list(range(len(df_new)))

        df_new.to_csv('./humor_datasets/twss/data.csv', index=False)

        train, test = train_test_split(df_new, test_size=0.2, shuffle=True)
        train.to_csv('./humor_datasets/twss/train.csv', index=False)
        test.to_csv('./humor_datasets/twss/test.csv', index=False)

    @staticmethod
    def load_igg():
        path = './original_datasets/igg/all_data.csv'
        df = pd.read_csv(path)

        df_new = pd.DataFrame()
        df_new['sentence'] = df['title']
        df_new['label'] = df['label']
        df_new['idx'] = df['id']
        df_new = df_new[df_new['sentence'].notna()]
        df_new = df_new.sample(frac=1, random_state=0)

        df_new.to_csv('./humor_datasets/igg/data.csv', index=False)

        train, test = train_test_split(df_new, test_size=0.2, shuffle=True)
        train.to_csv('./humor_datasets/igg/train.csv', index=False)
        test.to_csv('./humor_datasets/igg/test.csv', index=False)

    @staticmethod
    def split_test_to_val(path, dataset):
        full_path = path + dataset + '/'
        df = pd.read_csv(full_path + 'test.csv')
        df.to_csv(full_path + 'all_test.csv')
        split_at = int(len(df) / 2)
        test = df.iloc[:split_at]
        val = df.iloc[split_at:]
        test.to_csv(full_path + 'test.csv', ignore_index=True)
        val.to_csv(full_path + 'val.csv', ignore_index=True)

    @staticmethod
    def convert_data_to_T5(path, dataset):
        full_path = path + dataset + '/'
        for split in ['train', 'test', 'val']:
            df = pd.read_csv(full_path + split + '.csv')
            df['target'] = df['label']
            df['target'] = df['target'].apply(lambda x: 'funny' if x == 1 else 'not funny')
            os.makedirs(full_path + 'T5', exist_ok=True)
            df.to_csv(full_path + 'T5/' + split + '.csv', ignore_index=True)


if __name__ == '__main__':
    pass


    ## constructing datasets
    # DataPreprocessing.load_headlines()
    # DataPreprocessing.load_amazon()
    DataPreprocessing.load_twss()
    # DataPreprocessing.load_igg()

    ## balance train
    data_path = './humor_datasets/'
    # datasets_names = ['amazon', 'headlines', 'igg', 'twss']
    # datasets_names = ['twss']
    # DataPreprocessing.balance_train(data_path, datasets_names)
    # DataPreprocessing.split_test_to_val(data_path, 'amazon')
    DataPreprocessing.convert_data_to_T5(data_path, 'amazon')