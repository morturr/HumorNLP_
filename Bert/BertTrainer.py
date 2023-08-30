import evaluate
import logging
import wandb
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
from sklearn.metrics import accuracy_score

import sys
sys.path.append('../')
from Utils.utils import DataTrainingArguments, ModelArguments, DEVICE

logger = logging.getLogger(__name__)
wandb.init(project='HumorNLP')


class BertTrainer:
    def __init__(self):
        self.tokenizer = None
        self.metric = None
        self.config = None
        self.model = None
        self.classifier = None
        self.raw_datasets = None
        self.results = {}
        self.trainer = None
        self.dataset_columns = None
        self.text_column, self.label_column = None, None
        self.train_datasets, self.eval_datasets, self.predict_datasets = None, None, None
        self.train_idx = -1
        self.parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        self.model_args, self.data_args, self.training_args = self.parser.parse_args_into_dataclasses()

    def pipeline(self):
        set_seed(self.training_args.seed)
        self.load_files()
        self.config_model()
        self.set_data_attr()
        self.preprocess_datasets()
        self.train_and_predict()

    def load_files(self):
        if (self.training_args.do_eval and 'no_val' in self.data_args.split_type) or \
                (not self.training_args.do_eval and 'with_val' in self.data_args.split_type):
            logger.warning(
                "The use of eval set is not compatible"
            )
        if self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            self.raw_datasets = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
                cache_dir=self.model_args.cache_dir,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )
        else:
            data_files = {}
            if self.data_args.train_file is not None:
                data_files["train"] = self.data_args.train_file
                extension = self.data_args.train_file.split(".")[-1]
            if self.data_args.validation_file is not None:
                data_files["validation"] = self.data_args.validation_file
                extension = self.data_args.validation_file.split(".")[-1]
            if len(self.data_args.trained_on) > 1:
                for dataset in self.data_args.trained_on:
                    curr_train_path = self.data_args.train_path_template.format(dataset=dataset,
                                                                               split_type=self.data_args.split_type,
                                                                               split_name='train')
                    curr_val_path = self.data_args.train_path_template.format(dataset=dataset,
                                                                             split_type=self.data_args.split_type,
                                                                             split_name='val')
                    data_files[f'{dataset}_train'] = curr_train_path
                    data_files[f'{dataset}_validation'] = curr_val_path
                extension = self.data_args.train_path_template.split(".")[-1]
            if self.data_args.test_file is not None:
                data_files["test"] = self.data_args.test_file
                extension = self.data_args.test_file.split(".")[-1]
            if self.data_args.datasets_to_predict is not None:
                extension = self.data_args.test_path_template.split(".")[-1]
                for dataset in self.data_args.datasets_to_predict:
                    curr_predict_path = self.data_args.test_path_template.format(dataset=dataset,
                                                                                 split_type=self.data_args.split_type,
                                                                                 split_name='test')
                    data_files[f'{dataset}_test'] = curr_predict_path
            self.raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=self.model_args.cache_dir,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )

    def config_model(self):
        self.config = BertConfig.from_pretrained(self.model_args.model_name_or_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_args.model_name_or_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_args.model_name_or_path,
                                                                   config=self.config).to(DEVICE)
        self.classifier = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, device=0)
        self.metric = evaluate.load('accuracy')

    def set_data_attr(self):
        # column names for input/target
        self.dataset_columns = ('bert_sentence', 'label')

        if self.data_args.text_column is None:
            self.text_column = self.dataset_columns[0] if self.dataset_columns is not None else ''
        else:
            self.text_column = self.data_args.text_column

        if self.data_args.label_column is None:
            self.label_column = self.dataset_columns[1] if self.dataset_columns is not None else ''
        else:
            self.label_column = self.data_args.label_column

    def preprocess_function(self, data):
        result = self.tokenizer(data[self.text_column], truncation=True, max_length=512)
        return result

    def compute_metrics(self, p: EvalPrediction):
        predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        predictions = np.argmax(predictions, axis=1)
        result = self.metric.compute(predictions=predictions, references=p.label_ids)
        return result

    def preprocess_datasets(self):
        if self.training_args.do_train:
            train_dataset = self.raw_datasets["train"]
            if len(self.data_args.trained_on) > 1:
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
                        batched=True)
                    # remove_columns=self.train_datasets[i].column_names,  ## all columns??)

        if self.training_args.do_eval:
            eval_dataset = self.raw_datasets["validation"]
            if len(self.data_args.trained_on) > 1:
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
                        batched=True)
                    # remove_columns=self.eval_datasets[i].column_names)

        if self.training_args.do_predict:
            predict_dataset = self.raw_datasets["test"]
            if self.data_args.datasets_to_predict:
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
                        batched=True)
                    # remove_columns=self.predict_datasets[i].column_names)

    def train_and_predict(self):
        epochs = self.data_args.epochs
        batch_sizes = self.data_args.batch_sizes
        lrs = self.data_args.learning_rates
        seeds = self.data_args.seeds
        general_output_dir = self.training_args.output_dir

        for i in range(len(self.data_args.trained_on)):
            self.train_idx = i
            for ep in epochs:
                for bs in batch_sizes:
                    for lr in lrs:
                        for seed in seeds:
                            self.set_run_details(ep, bs, lr, seed, general_output_dir)
                            self.train()
                            self.eval()
                            self.predict()
                            self.compute_performance(ep, bs, lr, seed)
                            wandb.finish()

            self.save_results()

    def set_run_details(self, ep, bs, lr, seed, general_output_dir):
        self.training_args.num_train_epochs = ep
        self.training_args.per_device_train_batch_size = bs
        self.training_args.per_device_eval_batch_size = bs
        self.training_args.learning_rate = lr
        self.training_args.seed = seed
        time = datetime.now()
        self.training_args.run_name = '{model_name}-{split_type}_on_{trained_on}_seed={seed}_ep={ep}_bs={bs}' \
                                      '_lr={lr}_{date}_{hour}_{minute}'.format(
            model_name=self.model_args.model_name_or_path,
            date=time.date(), hour=time.hour, minute=time.minute,
            trained_on=self.data_args.trained_on[self.train_idx], seed=self.training_args.seed,
            ep=self.training_args.num_train_epochs,
            bs=self.training_args.per_device_train_batch_size,
            lr=self.training_args.learning_rate,
            split_type=self.data_args.split_type)

        wandb.init(project='HumorNLP', name=self.training_args.run_name)
        wandb.run.name = self.training_args.run_name
        self.training_args.output_dir = '{0}/{1}'.format(general_output_dir,
                                                         self.training_args.run_name)

    def train(self):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_datasets[self.train_idx] if self.training_args.do_train else None,
            eval_dataset=self.eval_datasets[self.train_idx] if self.training_args.do_eval else None,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer
        )

        if self.training_args.do_train:
            train_result = self.trainer.train()
            if self.model_args.save_model:
                self.trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            max_train_samples = (
                self.data_args.max_train_samples if self.data_args.max_train_samples is not None else len(
                    self.train_datasets[self.train_idx])
            )
            metrics["train_samples"] = min(max_train_samples, len(self.train_datasets[self.train_idx]))

            self.trainer.log_metrics("train", metrics)
            if self.model_args.save_metrics:
                self.trainer.save_metrics("train", metrics)
            if self.model_args.save_state:
                self.trainer.save_state()

    def eval(self):
        # Evaluation
        if self.training_args.do_eval:
            logger.info("*** Evaluate ***")
            if isinstance(self.eval_datasets[self.train_idx], dict):
                metrics = {}
                for eval_ds_name, eval_ds in self.eval_datasets[self.train_idx].items():
                    dataset_metrics = self.trainer.evaluate(eval_dataset=eval_ds,
                                                            metric_key_prefix=f"eval_{eval_ds_name}")
                    metrics.update(dataset_metrics)
            else:
                metrics = self.trainer.evaluate(metric_key_prefix="eval")
            max_eval_samples = self.data_args.max_eval_samples if self.data_args.max_eval_samples is not None else len(
                self.eval_datasets[self.train_idx])
            metrics["eval_samples"] = min(max_eval_samples, len(self.eval_datasets[self.train_idx]))

            self.trainer.log_metrics("eval", metrics)
            if self.model_args.save_metrics:
                self.trainer.save_metrics("eval", metrics)

    def predict(self):
        if self.training_args.do_predict:
            logger.info("*** Predict ***")
            for i in range(len(self.predict_datasets)):
                df_real = pd.read_csv(
                    f'../Data/humor_datasets/{self.data_args.datasets_to_predict[i]}/{self.data_args.split_type}/test.csv')
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
                df_pred['label'] = df['label'].apply(lambda s: s[-1])

                cols = ['id', 'bert_sentence', 'label']
                df_pred = df_pred[cols]

                os.makedirs(f'{self.training_args.output_dir}/predictions', exist_ok=True)
                output_prediction_file = os.path.join(
                    self.training_args.output_dir, 'predictions',
                    "{dataset}_preds.csv".format(dataset=self.data_args.datasets_to_predict[i]))
                df_pred.to_csv(output_prediction_file, index=False)

    def compute_performance(self, ep, bs, lr, seed):
        dataset_name = self.data_args.compute_on[self.train_idx]
        df_real = pd.read_csv(f'../Data/humor_datasets/{dataset_name}/{self.data_args.split_type}/test.csv')
        prediction_file = os.path.join(
            self.training_args.output_dir, 'predictions',
            "{dataset}_preds.csv".format(dataset=dataset_name))
        df_pred = pd.read_csv(prediction_file)
        df_real = df_real.iloc[:len(df_pred)]

        self.results[ep, bs, lr, seed] = accuracy_score(df_real.label, df_pred.label)

    def save_results(self):
        time = datetime.now()
        results_file_path = '../Data/output/results/{model_name}_on_{dataset}_{date}_{hour}_{minute}.txt'.format(
            model_name=self.model_args.model_name_or_path,
            dataset=self.data_args.trained_on[self.train_idx],
            date=time.date(),
            hour=time.hour, minute=time.minute
        )

        with open(results_file_path, 'a') as f:
            for k, v in self.results.items():
                f.write(f'ep: {k[0]}, bs: {k[1]}, lr: {k[2]}, seed: {k[3]}\n')
                f.write(f'accuracy = {v}\n')


if __name__ == '__main__':
    bert_trainer = BertTrainer()
    bert_trainer.pipeline()
