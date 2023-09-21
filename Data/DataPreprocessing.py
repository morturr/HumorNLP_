import os
import pandas as pd
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from itertools import combinations


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
    def convert_label_to_target(label):
        return 'funny' if label == 1 else 'not funny'

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
                test.to_csv(curr_path + 'before_balance/' + 'test.csv', index=False)
                train.to_csv(curr_path + 'before_balance/' + 'train.csv', index=False)
                test = test.append(larger_df.iloc[:labels_diff], ignore_index=True)
                larger_df = larger_df.iloc[labels_diff:]
                test = test.sample(frac=1, random_state=0, ignore_index=True)
                train = larger_df.append(shorter_df)
                train = train.sample(frac=1, random_state=0, ignore_index=True)
                test.to_csv(curr_path + 'test.csv', index=False)
                train.to_csv(curr_path + 'train.csv', index=False)

    @staticmethod
    def balance_train(train_df, test_df):
        train_label_1 = train_df[train_df['label'] == 1]
        train_label_0 = train_df[train_df['label'] == 0]
        larger_df = train_label_1 if len(train_label_1) > len(train_label_0) else train_label_0
        shorter_df = train_label_0 if len(train_label_1) > len(train_label_0) else train_label_1
        labels_diff = abs(len(train_label_1) - len(train_label_0))

        if labels_diff > 0:
            test_df = test_df.append(larger_df.iloc[:labels_diff], ignore_index=True)
            larger_df = larger_df.iloc[labels_diff:]
            train_df = larger_df.append(shorter_df)
            train_df = train_df.sample(frac=1, random_state=0, ignore_index=True)
            test_df = test_df.sample(frac=1, random_state=0, ignore_index=True)

        return train_df, test_df

    @staticmethod
    def balance_dataframe(df):
        df_label_1 = df[df['label'] == 1]
        df_label_0 = df[df['label'] == 0]
        larger_df = df_label_1 if len(df_label_1) > len(df_label_0) else df_label_0
        shorter_df = df_label_0 if len(df_label_1) > len(df_label_0) else df_label_1
        labels_diff = abs(len(df_label_1) - len(df_label_0))

        if labels_diff > 0:
            larger_df = larger_df.iloc[labels_diff:]
            new_df = larger_df.append(shorter_df)
            new_df = new_df.sample(frac=1, random_state=0, ignore_index=True)
            return new_df

        else:
            return df


    @staticmethod
    def split_train_test(df, path):
        os.makedirs(path, exist_ok=True)
        train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=0)
        train, test = DataPreprocessing.balance_train(train, test)
        test.to_csv(f'{path}/test.csv', index=False)
        train.to_csv(f'{path}/train.csv', index=False)

    @staticmethod
    def split_train_test_val(df, path):
        os.makedirs(path, exist_ok=True)
        train, test = train_test_split(df, test_size=0.3, shuffle=True, random_state=0)
        train, test = DataPreprocessing.balance_train(train, test)
        test, val = train_test_split(test, test_size=0.5, shuffle=True, random_state=0)
        test.to_csv(f'{path}/test.csv', index=False)
        train.to_csv(f'{path}/train.csv', index=False)
        val.to_csv(f'{path}/val.csv', index=False)

    @staticmethod
    def process_and_save_dataset(df_new, path):
        df_new = df_new.sample(frac=1, random_state=0)
        cols = ['id', 'bert_sentence', 't5_sentence', 'target', 'label']
        df_new = df_new[cols]

        os.makedirs(path, exist_ok=True)
        df_new.to_csv(f'{path}/data.csv', index=False)

        DataPreprocessing.split_train_test(df_new, f'{path}/no_val')
        DataPreprocessing.split_train_test_val(df_new, f'{path}/with_val')

    @staticmethod
    def preprocess_amazon():
        input_path = './original_datasets/amazon/all_data.csv'
        output_path = './humor_datasets/amazon'
        df = pd.read_csv(input_path)
        df_new = pd.DataFrame()
        df_new['bert_sentence'] = df['question'] + ' [SEP] ' + df['product_description']
        df_new['t5_sentence'] = df['question'] + ' </s> ' + df['product_description']
        df_new['label'] = df['label']
        df_new['target'] = df_new['label'].apply(DataPreprocessing.convert_label_to_target)
        df_new = df_new[df_new['bert_sentence'].notna()]
        df_new = df_new[df_new['t5_sentence'].notna()]
        df_new['id'] = range(0, len(df_new))

        DataPreprocessing.process_and_save_dataset(df_new, output_path)

    @staticmethod
    def preprocess_headlines():
        def edit_headline(row):
            headline = row['original']
            edit_word = row['edit']
            res = headline[:headline.index('<')] + edit_word + headline[headline.index('>') + 1:]
            return res

        input_path = './original_datasets/headlines'
        output_path = './humor_datasets/headlines'
        # output_path = './humor_datasets/headlines/sanity-check'
        origin_train = pd.read_csv(f'{input_path}/train.csv')
        origin_test = pd.read_csv(f'{input_path}/test.csv')
        df = origin_train.append(origin_test, ignore_index=True)

        df = df[df['edit'].notna()]
        df_new = pd.DataFrame()
        df_new['bert_sentence'] = df.apply(edit_headline, axis=1)
        df_new['t5_sentence'] = df_new['bert_sentence']
        df_new['label'] = df.apply(lambda row: 1 if row['meanGrade'] >= 1 else 0, axis=1)
        df_new['target'] = df_new['label'].apply(DataPreprocessing.convert_label_to_target)
        df_new['id'] = df['id']
        DataPreprocessing.process_and_save_dataset(df_new, output_path)

    @staticmethod
    def preprocess_twss():
        input_path = './original_datasets/twss/all_data.csv'
        output_path = './humor_datasets/twss'
        df = pd.read_csv(input_path)
        df = df[df['sentence'].notna()]
        df_new = pd.DataFrame()
        df_new['bert_sentence'] = df['sentence']
        df_new['t5_sentence'] = df_new['bert_sentence']
        df_new['label'] = df['label']
        df_new['id'] = df['idx']
        df_new['target'] = df_new['label'].apply(DataPreprocessing.convert_label_to_target)
        DataPreprocessing.process_and_save_dataset(df_new, output_path)

    @staticmethod
    def preprocess_igg():
        input_path = './original_datasets/igg/all_data.csv'
        output_path = './humor_datasets/igg'
        df = pd.read_csv(input_path)

        df_new = pd.DataFrame()
        df_new['bert_sentence'] = df['title']
        df_new['t5_sentence'] = df_new['bert_sentence']
        df_new['label'] = df['label']
        df_new['target'] = df_new['label'].apply(DataPreprocessing.convert_label_to_target)
        df_new['id'] = df['id']
        df_new = df_new[df_new['bert_sentence'].notna()]

        DataPreprocessing.process_and_save_dataset(df_new, output_path)

    @staticmethod
    def preprocess_datasets():
        DataPreprocessing.preprocess_amazon()
        DataPreprocessing.preprocess_headlines()
        DataPreprocessing.preprocess_igg()
        DataPreprocessing.preprocess_twss()

    @staticmethod
    def split_test_to_val(path, dataset):
        full_path = path + dataset + '/'
        df = pd.read_csv(full_path + 'test.csv')
        df.to_csv(full_path + 'all_test.csv')
        split_at = int(len(df) / 2)
        test = df.iloc[:split_at]
        val = df.iloc[split_at:]
        test.to_csv(full_path + 'test.csv', index=False)
        val.to_csv(full_path + 'val.csv', index=False)

    @staticmethod
    def convert_data_to_T5(path, datasets):
        for dataset in datasets:
            full_path = path + dataset + '/'
            for split in ['train', 'test']:#, 'val']:
                df = pd.read_csv(full_path + split + '.csv')
                df['target'] = df['label']
                df['target'] = df['target'].apply(lambda x: 'funny' if x == 1 else 'not funny')
                if 'Unnamed: 0' in df.columns:
                    df = df.drop(columns='Unnamed: 0')
                cols = ['sentence', 'label', 'idx', 'target']
                df = df[cols]
                os.makedirs(full_path + 'T5', exist_ok=True)
                df.to_csv(full_path + 'T5/' + split + '.csv', index=False)

    @staticmethod
    def convert_amazon_data_to_T5(path):
        full_path = path + 'amazon/T5' + '/'
        for split in ['train', 'test']:
            df = pd.read_csv(full_path + split + '.csv')
            df['sentence'] = df['sentence'].apply(lambda row: row.replace('[SEP]', '</s>'))
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns='Unnamed: 0')
            df.to_csv(full_path + split + '.csv', index=False)

    @staticmethod
    def create_fixed_size_train():
        datasets = ['amazon', 'headlines', 'igg', 'twss']
        data_path = './humor_datasets'

        # find the smallest train size
        train_size_no_val = DataPreprocessing.get_smallest_train_size(datasets, data_path, 'no_val')
        train_size_with_val = DataPreprocessing.get_smallest_train_size(datasets, data_path, 'with_val')

        DataPreprocessing.fix_train_size(data_path, datasets, 'no_val', train_size_no_val)
        DataPreprocessing.fix_train_size(data_path, datasets, 'with_val', train_size_with_val)

    @staticmethod
    def fix_train_size(data_path, datasets, split_type, train_size):
        for dataset in datasets:
            df = pd.read_csv(f'{data_path}/{dataset}/{split_type}/train.csv')
            df_label_1_train = df[df.label == 1].iloc[:int(train_size / 2)]
            df_label_0_train = df[df.label == 0].iloc[:int(train_size / 2)]
            df_label_1_rest = df[df.label == 1].iloc[int(train_size / 2):]
            df_label_0_rest = df[df.label == 0].iloc[int(train_size / 2):]
            df_train = df_label_0_train.append(df_label_1_train)
            df_train = df_train.sample(frac=1, random_state=0, ignore_index=True)
            df_rest = df_label_0_rest.append(df_label_1_rest)
            df_rest = df_rest.sample(frac=1, random_state=0, ignore_index=True)

            df_test = pd.read_csv(f'{data_path}/{dataset}/{split_type}/test.csv')
            df_val = None

            if split_type == 'no_val':
                df_test = df_test.append(df_rest)
            else:
                df_val = pd.read_csv(f'{data_path}/{dataset}/{split_type}/val.csv')
                df_test = df_test.append(df_rest.iloc[:int(len(df_rest) / 2)])
                df_val = df_val.append(df_rest.iloc[int(len(df_rest) / 2):])

            output_path = f'{data_path}/{dataset}/{split_type}_fixed_train'
            os.makedirs(output_path, exist_ok=True)
            df_train.to_csv(f'{output_path}/train.csv', index=False)
            df_test.to_csv(f'{output_path}/test.csv', index=False)
            if df_val is not None:
                df_val.to_csv(f'{output_path}/val.csv', index=False)

    @staticmethod
    def get_smallest_train_size(datasets, data_path, split_type):
        train_size = None
        for dataset in datasets:
            df = pd.read_csv(f'{data_path}/{dataset}/{split_type}/train.csv')
            if not train_size or train_size > len(df):
                train_size = len(df)
        return train_size

    @staticmethod
    def create_pair_datasets(split_name):
        def get_half_balanced_train(df):
            label_1 = df[df.label == 1]
            label_0 = df[df.label == 0]
            label_size = int(len(df) / 4)
            label_1 = label_1.iloc[:label_size]
            label_0 = label_0.iloc[:label_size]
            new_df = label_1.append(label_0, ignore_index=True)
            new_df = new_df.sample(frac=1, random_state=0, ignore_index=True)
            return new_df

        def get_fixed_size_val():
            min_val_size = None
            for dataset in datasets:
                df = pd.read_csv(f'{data_path}/{dataset}/{split_type}/val.csv')
                if min_val_size is None or len(df) < min_val_size:
                    min_val_size = len(df)

            return min_val_size

        datasets = ['amazon', 'headlines', 'igg', 'twss']
        data_path = './humor_datasets'
        split_type = 'with_val_fixed_train'
        output_path = data_path + '/paired_datasets/'
        pair_dataset = list(combinations(datasets, 2))
        dfs = {}

        for dataset in datasets:
            df = pd.read_csv(f'{data_path}/{dataset}/{split_type}/{split_name}.csv')
            if split_name == 'train':
                dfs[dataset] = get_half_balanced_train(df)
            elif split_name == 'val':
                size = get_fixed_size_val()
                dfs[dataset] = df.iloc[:size]

        for pair in pair_dataset:
            dataset1, dataset2 = pair[0], pair[1]
            df1 = dfs[dataset1]
            df2 = dfs[dataset2]
            merged_df = df1.append(df2, ignore_index=True)
            merged_df = merged_df.sample(frac=1, random_state=0, ignore_index=False)
            merged_path = output_path + f'{dataset1}_{dataset2}/{split_type}/'
            os.makedirs(merged_path, exist_ok=True)
            merged_df.to_csv(merged_path + f'{split_name}.csv', index=False)

    @staticmethod
    def balance_all_datasets():
        datasets = ['amazon', 'headlines', 'igg', 'twss']
        output_path = './humor_datasets/{dataset}/kfold_cv/'
        input_path = './humor_datasets/{dataset}/data.csv'

        for dataset in datasets:
            df = pd.read_csv(input_path.format(dataset=dataset))
            balanced_df = DataPreprocessing.balance_dataframe(df)
            curr_output_path = output_path.format(dataset=dataset)
            os.makedirs(curr_output_path, exist_ok=True)
            balanced_df.to_csv(curr_output_path + 'balanced_data.csv', index=False)

    @staticmethod
    def create_kfold_cv_data():
        def get_train_size():
            # compute fixed train size by igg train size
            igg_df = pd.read_csv(kfold_path.format(dataset='igg') + 'balanced_data.csv')
            splits = kfold.split(igg_df, igg_df['label'])
            for train, test in splits:
                fixed_train_size = len(train)
                if len(train) % 2 == 0:
                    break

            return fixed_train_size

        datasets = ['amazon', 'headlines', 'igg', 'twss']
        kfold_path = './humor_datasets/{dataset}/kfold_cv/'
        kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
        fixed_train_size = get_train_size()

        for dataset in datasets:
            df = pd.read_csv(kfold_path.format(dataset=dataset) + 'balanced_data.csv')
            for i, indices in enumerate(kfold.split(df, df['label'])):
                train_idxs, test_idxs = indices[0], indices[1]
                df_train = df.iloc[train_idxs]
                df_test = df.iloc[test_idxs]
                df_label_1 = df_train[df_train['label'] == 1].sample(n=int(fixed_train_size / 2), random_state=0)
                df_label_0 = df_train[df_train['label'] == 0].sample(n=int(fixed_train_size / 2), random_state=0)
                df_train = df_label_1.append(df_label_0, ignore_index=True)
                df_train = df_train.sample(frac=1, random_state=0)
                df_test, df_val = train_test_split(df_test, test_size=0.5, shuffle=True, random_state=0)
                curr_path = kfold_path.format(dataset=dataset) + f'fold_{i}/'

                os.makedirs(curr_path, exist_ok=True)
                df_train.to_csv(curr_path + 'train.csv', index=False)
                df_test.to_csv(curr_path + 'test.csv', index=False)
                df_val.to_csv(curr_path + 'val.csv', index=False)


if __name__ == '__main__':
    ## constructing datasets
    # DataPreprocessing.preprocess_datasets()
    # DataPreprocessing.create_fixed_size_train()
    # split_name = 'train'
    # split_name = 'val'
    # DataPreprocessing.create_pair_datasets(split_name)
    # split_name = 'train'
    # DataPreprocessing.create_pair_datasets(split_name)

    ## check headlines creation
    # DataPreprocessing.preprocess_headlines()
    # DataPreprocessing.create_fixed_size_train()
    # DataPreprocessing.balance_all_datasets()
    DataPreprocessing.create_kfold_cv_data()

