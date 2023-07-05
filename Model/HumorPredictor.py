from transformers import TextClassificationPipeline
import pandas as pd
import os
from Utils.utils import print_str, print_cur_time
from os.path import exists


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

    def predict(self, predict_on_dataset):
        print_str('STARTED PREDICT ON {0}'.format(predict_on_dataset))
        print_cur_time('before predict')
        self.preds[predict_on_dataset] = self.classifier(self.datasets[predict_on_dataset]['sentence'],
                                                         batch_size=1)
        print_cur_time('after predict')
        print_str('FINISHED PREDICT ON {0}'.format(predict_on_dataset))
        return self.preds[predict_on_dataset]

    def write_predictions(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        for dataset_name, preds in self.preds.items():
            preds_dict = pd.DataFrame.from_dict(preds)
            preds_dict['sentence'] = self.datasets[dataset_name]['sentence']
            preds_dict['label'] = preds_dict['label'].apply(lambda s: s[-1])
            preds_dict.to_csv(f'{path}/{dataset_name}_preds.csv')

    @staticmethod
    def compute_accuracy(pred_labels_path, test_labels_path):
        if exists(pred_labels_path) and exists(test_labels_path):
            preds = pd.read_csv(pred_labels_path)
            test = pd.read_csv(test_labels_path)
        equals = (test['label'] == preds['label']).sum()
        return equals / len(preds)


if __name__ == '__main__':
    dataset_names = ['amazon', 'headlines', 'twss', 'igg']
    data_path = '../Data/humor_datasets/'
    pred_path = '../Data/output/predictions/2023-06-19/'
    accuracies = {}
    for dataset in dataset_names:
        accuracies[dataset] = HumorPredictor.compute_accuracy(pred_path + f'{dataset}_preds.csv',
                                                              data_path + f'{dataset}/test.csv')

    print(accuracies)
