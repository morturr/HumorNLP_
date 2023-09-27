import evaluate
import logging
# import wandb
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    TextClassificationPipeline,
    EvalPrediction,
    set_seed,
    Trainer)

import numpy as np
import pandas as pd
from datasets import load_dataset
from datetime import datetime
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score

import sys
sys.path.append('../')
from Utils.utils import DEVICE, print_cur_time
from Model.HumorTrainer import HumorTrainer

logger = logging.getLogger(__name__)
# wandb.init(project='HumorNLP')


class BertTrainer(HumorTrainer):
    def __init__(self):
        self.classifier = None
        self.label_column = None

        HumorTrainer.__init__(self)

    def config_and_tokenizer(self):
        self.config = BertConfig.from_pretrained(self.model_args.model_name_or_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_args.model_name_or_path)
        self.metric = evaluate.load('accuracy')

    def init_model(self):
        self.model = BertForSequenceClassification.from_pretrained(self.model_args.model_name_or_path,
                                                                   config=self.config).to(DEVICE)
        self.classifier = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, device=0)

    def set_data_attr(self):
        # column names for input/target
        self.dataset_columns = ('bert_sentence', 'label')

        if self.data_args.label_column is None:
            self.label_column = self.dataset_columns[1] if self.dataset_columns is not None else ''
        else:
            self.label_column = self.data_args.label_column

        HumorTrainer.set_data_attr(self)

    def preprocess_function(self, data):
        result = self.tokenizer(data[self.text_column], truncation=True, max_length=512)
        return result

    def compute_metrics(self, p: EvalPrediction):
        predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        predictions = np.argmax(predictions, axis=1)
        result = self.metric.compute(predictions=predictions, references=p.label_ids)
        return result

    def preprocess_datasets(self):
        # HumorTrainer.preprocess_datasets(self, remove_columns=False)

        remove_columns = False

        if self.training_args.do_train:
            if self.data_args.train_file:
                train_dataset = self.raw_datasets["train"]
            if self.data_args.train_path_template is not None:
                print_cur_time('Loading train datasets from train_path_template')
                self.train_datasets = []
                for dataset in self.data_args.trained_on:
                    self.train_datasets.append(self.raw_datasets[f'{dataset}_train'])
            else:
                self.train_datasets = [train_dataset]
            for i in range(len(self.train_datasets)):
                if self.data_args.max_train_samples is not None:
                    max_train_samples = min(len(self.train_datasets[i]), self.data_args.max_train_samples)
                    self.train_datasets[i] = self.train_datasets[i].select(range(max_train_samples))
                with self.training_args.main_process_first(desc="train dataset map pre-processing"):
                    self.train_datasets[i] = self.train_datasets[i].map(
                        self.preprocess_function,
                        batched=True,
                        remove_columns=self.train_datasets[i].column_names if remove_columns else None)

        if self.training_args.do_eval:
            if self.data_args.validation_file:
                eval_dataset = self.raw_datasets["validation"]
            if self.data_args.train_path_template is not None:
                print_cur_time('Loading validation datasets from train_path_template')
                self.eval_datasets = []
                for dataset in self.data_args.trained_on:
                    self.eval_datasets.append(self.raw_datasets[f'{dataset}_validation'])
            else:
                self.eval_datasets = [eval_dataset]
            for i in range(len(self.eval_datasets)):
                if self.data_args.max_eval_samples is not None:
                    max_eval_samples = min(len(self.eval_datasets[i]), self.data_args.max_eval_samples)
                    self.eval_datasets[i] = self.eval_datasets[i].select(range(max_eval_samples))
                with self.training_args.main_process_first(desc="validation dataset map pre-processing"):
                    self.eval_datasets[i] = self.eval_datasets[i].map(
                        self.preprocess_function,
                        batched=True,
                        remove_columns=self.eval_datasets[i].column_names if remove_columns else None)

        if self.training_args.do_predict:
            if self.data_args.test_file:
                predict_dataset = self.raw_datasets["test"]
            if self.data_args.datasets_to_predict and \
                    self.data_args.test_path_template is not None:
                print_cur_time('Loading test datasets from test_path_template')
                self.predict_datasets = []
                for dataset in self.data_args.datasets_to_predict:
                    self.predict_datasets.append(self.raw_datasets[f'{dataset}_test'])
            else:
                self.predict_datasets = [predict_dataset]

            for i in range(len(self.predict_datasets)):
                if self.data_args.max_predict_samples is not None:
                    max_predict_samples = min(len(self.predict_datasets[i]), self.data_args.max_predict_samples)
                    self.predict_datasets[i] = self.predict_datasets[i].select(range(max_predict_samples))
                with self.training_args.main_process_first(desc="prediction dataset map pre-processing"):
                    self.predict_datasets[i] = self.predict_datasets[i].map(
                        self.preprocess_function,
                        batched=True,
                        remove_columns=self.predict_datasets[i].column_names if remove_columns else None)

    def train(self):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_datasets[self.train_idx] if self.training_args.do_train else None,
            eval_dataset=self.eval_datasets[self.train_idx] if self.training_args.do_eval else None,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer
        )

        HumorTrainer.train(self)

    def predict(self, ep, bs, lr, seed):
        if self.training_args.do_predict:
            logger.info("*** Predict ***")
            print_cur_time(f'START PREDICT OF {self.training_args.run_name}')

            for i in range(len(self.predict_datasets)):
                print_cur_time(f'START PREDICT ON {self.data_args.datasets_to_predict[i]}')
                # df_real = pd.read_csv(
                #     f'../Data/humor_datasets/{self.data_args.datasets_to_predict[i]}/{self.data_args.split_type}/test.csv')

                if self.data_args.test_path_template:
                    df_real_path = self.data_args.test_path_template.format(
                        dataset=self.data_args.datasets_to_predict[i], split_type=self.data_args.split_type, split_name='test'
                    )

                else:
                    df_real_path = self.data_args.test_file

                print_cur_time(f'PREDICT file path  {df_real_path}')

                df_real = pd.read_csv(df_real_path)
                max_predict_samples = (
                    min(self.data_args.max_predict_samples, len(df_real))
                    if self.data_args.max_predict_samples is not None else
                    len(df_real)
                )
                df_real = df_real.iloc[list(range(max_predict_samples))]

                predictions = self.classifier(df_real['bert_sentence'].to_list())  # , batch_size=1)
                df = pd.DataFrame.from_dict(predictions)
                df_pred = pd.DataFrame()
                df_pred['bert_sentence'] = df_real['bert_sentence']
                df_pred['id'] = df_real['id']
                df_pred['label'] = df['label'].apply(lambda s: int(s[-1]))
                df_pred['true_label'] = df_real['label']

                cols = ['id', 'bert_sentence', 'label', 'true_label']
                df_pred = df_pred[cols]

                accuracy = float("%.4f" % accuracy_score(df_real.label, df_pred.label))
                recall = float("%.4f" % recall_score(df_real.label, df_pred.label))
                precision = float("%.4f" % precision_score(df_real.label, df_pred.label))

                row_to_final_results = {'model': self.model_args.model_name_or_path,
                                        'trained_on': self.data_args.trained_on[self.train_idx],
                                        'epoch': ep, 'batch_size': bs,
                                        'learning_rate': lr, 'seed': seed,
                                        'predict_on': self.data_args.datasets_to_predict[i],
                                        'accuracy': accuracy, 'recall': recall, 'precision': precision}

                print(row_to_final_results)

                self.final_results_df = self.final_results_df.append([row_to_final_results])

                os.makedirs(f'{self.training_args.output_dir}/predictions', exist_ok=True)
                output_prediction_file = os.path.join(
                    self.training_args.output_dir, 'predictions',
                    "{dataset}_preds.csv".format(dataset=self.data_args.datasets_to_predict[i]))
                df_pred.to_csv(output_prediction_file, index=False)

                print_cur_time(f'END PREDICT ON {self.data_args.datasets_to_predict[i]}')

            print_cur_time(f'END PREDICT OF {self.training_args.run_name}')


if __name__ == '__main__':
    bert_trainer = BertTrainer()
    bert_trainer.pipeline()
