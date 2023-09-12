import os

import pandas as pd
import numpy as np


class DataAnalysis:
    def __init__(self):
        self.kfold_data_path = './humor_datasets/{dataset}/kfold_cv/'
        self.datasets = ['amazon', 'headlines', 'igg', 'twss']
        self.num_k_folds = 4

    def pipeline(self):
        self.sentence_length_stats()

    def sentence_length_stats(self):
        def get_sentence_len_mean_std(lengths):
            return np.mean(lengths), np.std(lengths)

        sentence_lengths = {}
        df_stats = pd.DataFrame(columns=['dataset', 'fold', 'split', 'label', 'mean', 'std'])

        for dataset in self.datasets:
            for i in range(self.num_k_folds):
                curr_path = self.kfold_data_path.format(dataset=dataset) + f'fold_{i}/'
                train_df = pd.read_csv(curr_path + 'train.csv')
                test_df = pd.read_csv(curr_path + 'test.csv')

                train_df['sentence_length'] = train_df['t5_sentence'].apply(len)
                test_df['sentence_length'] = test_df['t5_sentence'].apply(len)

                train_sent_label_1 = train_df[train_df['label'] == 1]['sentence_length']
                train_sent_label_0 = train_df[train_df['label'] == 0]['sentence_length']
                test_sent_label_1 = test_df[test_df['label'] == 1]['sentence_length']
                test_sent_label_0 = test_df[test_df['label'] == 0]['sentence_length']

                for l in [0, 1]:
                    train_sent_label = train_df[train_df['label'] == l]['sentence_length']
                    test_sent_label = test_df[test_df['label'] == l]['sentence_length']

                    sentence_lengths[dataset, f'fold_{i}', 'train', f'label_{l}'] = \
                        get_sentence_len_mean_std(train_sent_label)

                    train_length_stat = get_sentence_len_mean_std(train_sent_label)
                    train_row_to_df = {'dataset': dataset, 'fold': i, 'split': 'train', 'label': l,
                                       'mean': train_length_stat[0], 'std': train_length_stat[1]}

                    test_length_stat = get_sentence_len_mean_std(test_sent_label)
                    test_row_to_df = {'dataset': dataset, 'fold': i, 'split': 'test', 'label': l,
                                      'mean': test_length_stat[0], 'std': test_length_stat[1]}

                    df_stats = df_stats.append([train_row_to_df, test_row_to_df], ignore_index=True)
                    sentence_lengths[dataset, f'fold_{i}', 'test', f'label_{l}'] = \
                        get_sentence_len_mean_std(test_sent_label)

        os.makedirs('./data_analysis/', exist_ok=True)
        df_stats.to_csv('./data_analysis/sentence_length.csv', index=False)

        df_results = pd.DataFrame(columns=['dataset', 'split', 'label', 'mean_of_means',
                                           'std_of_means', 'mean_of_stds', 'std_of_stds'])

        for dataset in self.datasets:
            dataset_sent_len = df_stats[df_stats['dataset'] == dataset]
            for split in ['train', 'test']:
                for label in [0, 1]:
                    curr_df = dataset_sent_len[(dataset_sent_len['split'] == split) &
                                               (dataset_sent_len['label'] == label)]
                    result_to_df = {'dataset': dataset, 'split': split, 'label': label,
                                    'mean_of_means': np.mean(curr_df['mean']),
                                    'std_of_means': np.std(curr_df['mean']),
                                    'mean_of_stds': np.mean(curr_df['std']),
                                    'std_of_stds': np.std(curr_df['std'])}
                    df_results = df_results.append([result_to_df])

        df_results.to_csv('./data_analysis/sentence_length_summarized.csv', index=False)


if __name__ == '__main__':
    da = DataAnalysis()
    da.pipeline()
