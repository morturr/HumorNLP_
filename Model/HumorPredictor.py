from transformers import TextClassificationPipeline
import pandas as pd
import numpy as np
import os
import sys

sys.path.append('../')

from Utils.utils import print_str, print_cur_time
from os.path import exists
from sklearn.metrics import precision_score, recall_score


# ===============================      Global Variables:      ===============================


# ====================================      Class:      ====================================
class HumorPredictor:
    def __init__(self, model, datasets, tokenizer):
        self.model = model
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.preds = {}
        self.classifier = self.init_classifier()

    def init_classifier(self):
        return TextClassificationPipeline(model=self.model.to('cpu'), tokenizer=self.tokenizer)

    def predict(self, predict_on_datasets):
        for predict_on_dataset in predict_on_datasets:
            print_str('STARTED PREDICT ON {0}'.format(predict_on_dataset))
            print_cur_time('before predict')
            self.preds[predict_on_dataset] = self.classifier(self.datasets[predict_on_dataset]['sentence'],
                                                             batch_size=1)
            print_cur_time('after predict')
            print_str('FINISHED PREDICT ON {0}'.format(predict_on_dataset))

    def write_predictions(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        for dataset_name, preds in self.preds.items():
            preds_dict = pd.DataFrame.from_dict(preds)
            preds_dict['sentence'] = self.datasets[dataset_name]['sentence']
            preds_dict['label'] = preds_dict['label'].apply(lambda s: s[-1])
            preds_dict.to_csv(f'{path}/{dataset_name}_preds.csv')

    def write_dataset_predictions(self, path, datasets_name):
        for dataset_name in datasets_name:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            preds = self.preds[dataset_name]
            preds_dict = pd.DataFrame.from_dict(preds)
            preds_dict['sentence'] = self.datasets[dataset_name]['sentence']
            preds_dict['label'] = preds_dict['label'].apply(lambda s: s[-1])
            preds_dict.to_csv(f'{path}/{dataset_name}_preds.csv')

    @staticmethod
    def compute_accuracy(pred_y, true_y):
        equals = (true_y['label'] == pred_y['label']).sum()
        return equals / len(pred_y)

    @staticmethod
    def compute_recall(df_pred, df_real):
        return recall_score(df_real['label'], df_pred['label'])

    @staticmethod
    def compute_precision(df_pred, df_real):
        return precision_score(df_real['label'], df_pred['label'])

    @staticmethod
    def save_performance():
        def get_run_details(run_name):
            run_data = run_name.split('_')
            model = run_data[0]
            dataset_name = run_data[2]
            seed = run_data[3][run_data[3].index('=') + 1:]

            return model, dataset_name, float(seed)

        output_path = '../Data/output/results/'
        # dataset_names = ['amazon', 'headlines', 'twss', 'igg']
        dataset_names = ['amazon', 'headlines', 'igg', 'twss']
        data_path = '../Data/humor_datasets/'
        models_name = [
                        # 'bert_on_amazon_seed=0',
                        # 'bert_on_headlines_seed=3',
                        # 'bert_on_igg_seed=28',
                        # 'bert_on_twss_seed=42',
                       # 'T5-with-val_on_amazon_seed=0',
                       # 'T5-with-val_on_headlines_seed=0',
                       # 'T5-with-val_on_igg_seed=42',
                       # 'T5-with-val_on_twss_seed=42',
                       #  'T5-no-val_on_amazon_seed=42',
                       #  'T5-no-val_on_headlines_seed=0',
                       #  'T5-no-val_on_igg_seed=42',
                       #  'T5-no-val_on_twss_seed=42',
                        'T5-no-val_on_amazon_seed=42_lr=5e-5'
                       ]

        df = pd.read_excel(output_path + 'humor_results_template.xlsx')
        df.fillna(method='ffill', axis=0, inplace=True)
        df.set_index(['performance', 'model', 'trained on', 'seed'], inplace=True)

        for model_name in models_name:
            base_model, dataset_name, seed = get_run_details(model_name)
            pred_path = '../Model/SavedModels/' + model_name + '/predictions/'
            accuracies = {}
            recall = {}
            precision = {}
            for dataset in dataset_names:
                pred_labels_path = pred_path + f'{dataset}_preds.csv'
                test_labels_path = data_path + f'{dataset}/T5/test.csv'
                # test_labels_path = data_path + f'{dataset}/partial_test.csv'
                if not (exists(pred_labels_path) and exists(test_labels_path)):
                    print('didnt find preds/test path')
                    continue

                _preds = pd.read_csv(pred_labels_path)
                _test = pd.read_csv(test_labels_path)
                if (len(_preds[_preds.label == -1]) > 0):
                    illegal_indices = _preds[_preds.label == -1].index
                    print(f'there are {len(illegal_indices)} illegal indices in {dataset_name} predictions on {dataset}')
                    _preds = _preds.drop(labels=illegal_indices, axis=0)
                    _test = _test.drop(labels=illegal_indices, axis=0)
                accuracies[dataset] = HumorPredictor.compute_accuracy(_preds, _test)
                recall[dataset] = HumorPredictor.compute_recall(_preds, _test)
                precision[dataset] = HumorPredictor.compute_precision(_preds, _test)

            print_str(f'performance for {model_name}')
            print(f'accuracies = {accuracies}')
            print(f'recall = {recall}')
            print(f'precision = {precision}')

            df.loc[('accuracy', base_model, dataset_name, seed)] = accuracies
            df.loc[('recall', base_model, dataset_name, seed)] = recall
            df.loc[('precision', base_model, dataset_name, seed)] = precision

        # save performance to output file
        i = 0
        while os.path.exists(output_path + f'humor_results_{i}.xlsx'):
            i += 1

        df.to_excel(output_path + f'humor_results_{i}.xlsx')

    @staticmethod
    def convert_T5_preds():
        def edit_row(row):
            if row['original'] not in ['funny', 'not funny']:
                row['edited'] = True
                if 'not' in row['original']:
                    row['target'] = 'not funny'
                    row['label'] = 0
                elif 'funny' in row['original']:
                    row['target'] = 'funny'
                    row['label'] = 1
                else:
                    row['target'] = 'illegal'
                    row['label'] = -1
            elif row['original'] == 'funny':
                row['label'] = 1
            elif row['original'] == 'not funny':
                row['label'] = 0
            return row

        datasets = ['amazon', 'headlines', 'igg', 'twss']
        model_preds_path = [
            # 'T5-no-val_on_amazon_seed=42',
            # 'T5-no-val_on_headlines_seed=0',
            # 'T5-no-val_on_igg_seed=42',
            # 'T5-no-val_on_twss_seed=42',
            'T5-no-val_on_amazon_seed=42_lr=5e-5'
        ]
        for model in model_preds_path:
            for dataset in datasets:
                f = open(f'SavedModels/{model}/predictions/{dataset}_generated_predictions.txt')
                data = f.read().split('\n')
                data = list(map(lambda x: x.strip(), data))

                df = pd.DataFrame()
                df['original'] = data
                df['target'] = df['original']
                df['edited'] = False

                df = df.apply(edit_row, axis=1)
                df_pred = df

                df_real = pd.read_csv(f'../Data/humor_datasets/{dataset}/T5/test.csv')
                df_pred['sentence'] = df_real['sentence']

                cols = ['sentence', 'target', 'label', 'original', 'edited']
                df_pred = df_pred[cols]

                df_pred.to_csv(f'SavedModels/{model}/predictions/{dataset}_preds.csv', index=False)

                all_count, legal_count = len(data), len(df_pred[df_pred['target'] != 'illegal'])
                # print(f'length of {dataset} is {all_count}')
                # print(f'length of legals in {dataset} is {legal_count}')
                # print(f'% legals = {100 * (legal_count / all_count)}')

                # df_real = pd.read_csv(f'../Data/humor_datasets/{dataset}/T5/test.csv')
                # print(f'accuracy of {dataset} is {compute_accuracy(df_pred, df_real)}')

if __name__ == '__main__':
    HumorPredictor.save_performance()
    # HumorPredictor.convert_T5_preds()