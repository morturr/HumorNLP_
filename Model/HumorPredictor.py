from transformers import TextClassificationPipeline
import pandas as pd
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
    def compute_recall(pred_y, true_y):
        return recall_score(true_y['label'], pred_y['label'], average='binary')

    @staticmethod
    def compute_precision(pred_y, true_y):
        return precision_score(true_y['label'], pred_y['label'], average='binary')

    @staticmethod
    def save_performance():
        def get_run_details(run_name):
            model, run_name = run_name[:run_name.index('_')], run_name[run_name.index('_') + 1:]
            run_name = run_name[run_name.index('_') + 1:]
            dataset_name, seed = run_name[:run_name.index('_')], run_name[run_name.index('=') + 1:]

            return model, dataset_name, float(seed)

        output_path = '../Data/output/results/'
        # dataset_names = ['amazon', 'headlines', 'twss', 'igg']
        dataset_names = ['amazon', 'headlines', 'igg', 'twss']
        data_path = '../Data/humor_datasets/'
        models_name = ['bert_on_amazon_seed=0',
                       'bert_on_headlines_seed=3',
                       'bert_on_igg_seed=28',
                       'bert_on_twss_seed=42']

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
                test_labels_path = data_path + f'{dataset}/test.csv'
                if not (exists(pred_labels_path) and exists(test_labels_path)):
                    print('didnt find preds/test path')
                    continue

                _preds = pd.read_csv(pred_labels_path)
                _test = pd.read_csv(test_labels_path)
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


if __name__ == '__main__':
    HumorPredictor.save_performance()
