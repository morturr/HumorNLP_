import logging
import os
import sys
# import wandb
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset
from filelock import FileLock
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from transformers.utils import is_offline_mode

import sys

sys.path.append('../')
from Utils.utils import DataTrainingArguments, ModelArguments, print_cur_time
from Model.HumorTrainer import HumorTrainer

logger = logging.getLogger(__name__)
from abc import ABC, abstractmethod


class HumorTrainer(ABC):
    def __init__(self):
        self.trainer = None
        self.metric = None
        self.text_column = None
        self.dataset_columns = None
        self.raw_datasets = None
        self.config, self.tokenizer, self.model = None, None, None
        self.train_datasets, self.eval_datasets, self.predict_datasets = None, None, None
        self.train_idx = -1
        self.results = {}
        self.results_df = None
        self.final_results_df = None

        self.parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
        self.model_args, self.data_args, self.training_args = self.parser.parse_args_into_dataclasses()
        self.run_dir_name = None

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    def pipeline(self):
        print_cur_time('STARTING PIPELINE')
        set_seed(self.training_args.seed)
        self.init_run_dir()
        self.load_files()
        self.config_and_tokenizer()
        self.set_data_attr()
        self.preprocess_datasets()
        self.train_and_predict()
        print_cur_time('ENDING PIPELINE')

    def init_run_dir(self):
        time = datetime.now()
        self.run_dir_name = '../Data/output/results/runs/{model_name}_{date}_{hour}_{minute}/'.format(
            model_name=self.model_args.model_name_or_path,
            date=time.date(),
            hour=time.hour, minute=time.minute
        )
        os.makedirs(self.run_dir_name, exist_ok=True)

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
            if self.data_args.train_path_template is not None:
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
            if self.data_args.datasets_to_predict is not None and \
                self.data_args.test_path_template is not None:
                extension = self.data_args.test_path_template.split(".")[-1]
                for dataset in self.data_args.datasets_to_predict:
                    curr_predict_path = self.data_args.test_path_template.format(dataset=dataset,
                                                                                 split_type=self.data_args.split_type,
                                                                                 split_name='test')
                    data_files[f'{dataset}_test'] = curr_predict_path

            print_cur_time('Loading datasets:')
            print(data_files)
            self.raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=self.model_args.cache_dir,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )

    @abstractmethod
    def config_and_tokenizer(self):
        pass

    @abstractmethod
    def init_model(self):
        pass

    def set_data_attr(self):
        if self.data_args.text_column is None:
            self.text_column = self.dataset_columns[0] if self.dataset_columns is not None else ''
        else:
            self.text_column = self.data_args.text_column

    @abstractmethod
    def preprocess_function(self):
        pass

    @abstractmethod
    def compute_metrics(self):
        pass

    def preprocess_datasets(self, remove_columns):
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
                        remove_columns=self.eval_datasets_datasets[i].column_names if remove_columns else None)

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

    def train_and_predict(self):
        print_cur_time('STARTING TRAINING')

        epochs = self.data_args.epochs
        batch_sizes = self.data_args.batch_sizes
        lrs = self.data_args.learning_rates
        seeds = self.data_args.seeds
        general_output_dir = self.training_args.output_dir

        self.results_df = pd.DataFrame(columns=['dataset', 'epoch', 'batch_size', 'learning_rate',
                                                'seed', 'accuracy'])

        self.final_results_df = pd.DataFrame(columns=['model', 'trained_on', 'epoch', 'batch_size',
                                                      'learning_rate', 'seed', 'predict_on', 'accuracy',
                                                      'recall', 'precision'])

        for i in range(len(self.data_args.trained_on)):
            self.train_idx = i
            # init results for this dataset
            self.results = {}

            for ep in epochs:
                for bs in batch_sizes:
                    for lr in lrs:
                        for seed in seeds:
                            self.set_run_details(ep, bs, lr, seed, general_output_dir)
                            self.init_model()
                            self.train()
                            self.eval()
                            self.predict(ep, bs, lr, seed)
                            self.compute_performance(ep, bs, lr, seed)
                            self.results_df.to_csv(self.run_dir_name + 'models_accuracy.csv', index=False)
                            self.final_results_df.to_csv(self.run_dir_name + 'models_performance_all.csv', index=False)

                            # wandb.finish()

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

        # wandb.init(project='HumorNLP', name=self.training_args.run_name)
        # wandb.run.name = self.training_args.run_name
        self.training_args.output_dir = '{0}/{1}'.format(general_output_dir,
                                                         self.training_args.run_name)

    def train(self):
        # Training
        if self.training_args.do_train:
            print_cur_time(f'START TRAIN ON {self.training_args.run_name}')

            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            # elif last_checkpoint is not None:
            #     checkpoint = last_checkpoint
            train_result = self.trainer.train(resume_from_checkpoint=checkpoint)

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

            print_cur_time(f'END TRAIN ON {self.training_args.run_name}')

    def eval(self):
        # Evaluation
        if self.training_args.do_eval:
            logger.info("*** Evaluate ***")
            print_cur_time(f'START EVAL ON {self.training_args.run_name}')

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

            print_cur_time(f'END EVAL ON {self.training_args.run_name}')

    @abstractmethod
    def predict(self):
        pass

    # TODO edit this function and do it in a good way! its just cause i dont feel comfortable
    def compute_performance(self, ep, bs, lr, seed):
        print_cur_time(f'START COMPUTE OF {self.training_args.run_name}')
        dataset_name = self.data_args.compute_on[self.train_idx]
        print_cur_time(f'COMPUTE dataset {dataset_name}')

        # df_real = pd.read_csv(f'../Data/humor_datasets/{dataset_name}/{self.data_args.split_type}/test.csv')
        if self.data_args.test_path_template:
            df_real_path = self.data_args.test_path_template.format(
                dataset=dataset_name, split_type=self.data_args.split_type, split_name='test'
            )

        else:
            df_real_path = self.data_args.test_file

        print_cur_time(f'COMPUTE file path  {df_real_path}')

        df_real = pd.read_csv(df_real_path)
        prediction_file = os.path.join(
            self.training_args.output_dir, 'predictions',
            "{dataset}_preds.csv".format(dataset=dataset_name))
        df_pred = pd.read_csv(prediction_file)
        total_count, legal_count = len(df_pred), len(df_pred)
        df_real = df_real.iloc[:total_count]

        # because from t5 we may get illegal predictions, marked by -1
        if len(df_pred[df_pred.label == -1]) > 0:
            illegal_indices = df_pred[df_pred.label == -1].index
            print(f'there are {len(illegal_indices)} illegal indices in {dataset_name} predictions on itself')
            df_pred = df_pred.drop(labels=illegal_indices, axis=0)
            df_real = df_real.drop(labels=illegal_indices, axis=0)
            legal_count = len(df_pred)

        percent_legal = (legal_count / total_count) * 100
        accuracy = accuracy_score(df_real.label, df_pred.label)
        self.results[ep, bs, lr, seed] = accuracy, percent_legal

        result_to_df = {'dataset': self.data_args.trained_on[self.train_idx],
                        'epoch': ep, 'batch_size': bs, 'learning_rate': lr,
                        'seed': seed, 'accuracy': accuracy}
        self.results_df = self.results_df.append([result_to_df], ignore_index=True)

        print_cur_time(f'END COMPUTE OF {self.training_args.run_name}')

    def save_results(self):
        time = datetime.now()
        results_file_path = '{run_dir}/{model_name}_on_{dataset}_{hour}_{minute}.txt'.format(
            model_name=self.model_args.model_name_or_path,
            dataset=self.data_args.trained_on[self.train_idx],
            run_dir=self.run_dir_name,
            hour=time.hour, minute=time.minute
        )

        with open(results_file_path, 'a') as f:
            for k, v in self.results.items():
                f.write(f'ep: {k[0]}, bs: {k[1]}, lr: {k[2]}, seed: {k[3]}\n')
                f.write(f'accuracy = {v[0]} on {v[1]}% legal \n')
