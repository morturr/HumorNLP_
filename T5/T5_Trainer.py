import logging
import os
import sys
import wandb
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset
from filelock import FileLock
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score

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
from Utils.utils import DataTrainingArguments, ModelArguments

logger = logging.getLogger(__name__)
wandb.init(project='HumorNLP')

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


class T5_Trainer:
    def __init__(self):
        self.trainer = None
        self.padding, self.max_target_length, self.prefix = None, None, None
        self.data_collator, self.metric = None, None
        self.target_column, self.text_column, self.dataset_columns = None, None, None
        self.raw_datasets = None
        self.config, self.tokenizer, self.model = None, None, None
        self.train_datasets, self.eval_datasets, self.predict_datasets = None, None, None
        self.train_idx = -1
        self.results = {}

        self.parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
        self.model_args, self.data_args, self.training_args = self.parser.parse_args_into_dataclasses()

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

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
            self.raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=self.model_args.cache_dir,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )

    def config_model(self):
        self.config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            # use_fast=model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=self.config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Data collator
        label_pad_token_id = -100 if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )

        # Metric
        self.metric = evaluate.load("rouge")

    def set_data_attr(self):
        self.prefix = self.data_args.source_prefix if self.data_args.source_prefix is not None else ""

        # column names for input/target
        self.dataset_columns = ('t5_sentence', 'target')
        if self.data_args.text_column is None:
            self.text_column = self.dataset_columns[0] if self.dataset_columns is not None else ''
        else:
            self.text_column = self.data_args.text_column

        if self.data_args.target_column is None:
            self.target_column = self.dataset_columns[1] if self.dataset_columns is not None else ''
        else:
            self.target_column = self.data_args.target_column

        # Temporarily set max_target_length for training.
        self.max_target_length = self.data_args.max_target_length
        self.padding = "max_length" if self.data_args.pad_to_max_length else False

        # Override the decoding parameters of Seq2SeqTrainer
        self.training_args.generation_max_length = (
            self.training_args.generation_max_length
            if self.training_args.generation_max_length is not None
            else self.data_args.val_max_target_length
        )

    def preprocess_function(self, examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[self.text_column])):
            if examples[self.text_column][i] and examples[self.target_column][i]:
                inputs.append(examples[self.text_column][i])
                targets.append(examples[self.target_column][i])

        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs, max_length=self.data_args.max_source_length, padding=self.padding,
                                      truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = self.tokenizer(text_target=targets, max_length=self.max_target_length, padding=self.padding,
                                truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_datasets(self):
        if self.training_args.do_train:
            train_dataset = self.raw_datasets["train"]
            if self.data_args.train_path_template is not None:
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
                        # num_proc=self.data_args.preprocessing_num_workers,
                        remove_columns=self.train_datasets[i].column_names,  ## all columns??
                        # load_from_cache_file=not self.data_args.overwrite_cache,
                        desc="Running tokenizer on train dataset",
                    )

        if self.training_args.do_eval:
            eval_dataset = self.raw_datasets["validation"]
            if self.data_args.train_path_template is not None:
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
                        # num_proc=self.data_args.preprocessing_num_workers,
                        remove_columns=self.eval_datasets[i].column_names,
                        # load_from_cache_file=not self.data_args.overwrite_cache,
                        desc="Running tokenizer on validation dataset",
                    )

        if self.training_args.do_predict:
            if self.data_args.test_file:
                predict_dataset = self.raw_datasets["test"]
            if self.data_args.datasets_to_predict and \
                    self.data_args.test_path_template:
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
                        # num_proc=self.data_args.preprocessing_num_workers,
                        remove_columns=self.predict_datasets[i].column_names,
                        # load_from_cache_file=not self.data_args.overwrite_cache,
                        desc="Running tokenizer on prediction dataset",
                    )

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = T5_Trainer.postprocess_text(decoded_preds, decoded_labels)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

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
        # Initialize our Trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_datasets[self.train_idx] if self.training_args.do_train else None,
            eval_dataset=self.eval_datasets[self.train_idx] if self.training_args.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics if self.training_args.predict_with_generate else None,
        )

        # Training
        if self.training_args.do_train:
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
        # Predict
        if self.training_args.do_predict:
            logger.info("*** Predict ***")
            for i in range(len(self.predict_datasets)):
                predict_results = self.trainer.predict(self.predict_datasets[i], metric_key_prefix="predict")
                metrics = predict_results.metrics
                max_predict_samples = (
                    min(self.data_args.max_predict_samples, len(self.predict_datasets[i]))
                    if self.data_args.max_predict_samples is not None else
                    len(self.predict_datasets[i])
                )
                metrics["predict_samples"] = min(max_predict_samples, len(self.predict_datasets[i]))

                self.trainer.log_metrics("predict", metrics)
                if self.model_args.save_metrics:
                    self.trainer.save_metrics("predict", metrics)

                if self.trainer.is_world_process_zero():
                    if self.training_args.predict_with_generate:
                        predictions = predict_results.predictions
                        predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
                        predictions = self.tokenizer.batch_decode(
                            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                        )
                        predictions = [pred.strip() for pred in predictions]
                        self.save_predictions(predictions, self.data_args.datasets_to_predict[i])

                    else:
                        all_tokens = predict_results.predictions[0]
                        predicted_tokens = [[np.argmax(x) for x in tokens] for tokens in all_tokens]
                        predicted_tokens = [[token for token in tokens if token not in self.tokenizer.all_special_ids]
                                            for tokens in predicted_tokens]
                        predictions = [self.tokenizer.decode(tokens) for tokens in predicted_tokens]

                        self.save_predictions(predictions, self.data_args.datasets_to_predict[i])

    def save_predictions(self, predictions, predict_dataset):
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

        df = pd.DataFrame()
        df['original'] = predictions
        df['target'] = df['original']
        df['edited'] = False

        df = df.apply(edit_row, axis=1)
        df_pred = df

        # df_real = pd.read_csv(f'../Data/humor_datasets/{predict_dataset}/{self.data_args.split_type}/test.csv')
        df_real = pd.read_csv(self.data_args.test_path_template.format(
            dataset=predict_dataset, split_type=self.data_args.split_type, split_name='test'
        ))
        max_predict_samples = (
            min(self.data_args.max_predict_samples, len(df_real))
            if self.data_args.max_predict_samples is not None else
            len(df_real)
        )

        df_real = df_real.iloc[list(range(max_predict_samples))]
        df_pred['t5_sentence'] = df_real['t5_sentence']
        df_pred['id'] = df_real['id']
        cols = ['id', 't5_sentence', 'target', 'label', 'original', 'edited']
        df_pred = df_pred[cols]

        os.makedirs(f'{self.training_args.output_dir}/predictions', exist_ok=True)
        output_prediction_file = os.path.join(
            self.training_args.output_dir, 'predictions',
            "{dataset}_preds.csv".format(dataset=predict_dataset))
        df_pred.to_csv(output_prediction_file, index=False)

    # TODO edit this function and do it in a good way! its just cause i dont feel comfortable
    def compute_performance(self, ep, bs, lr, seed):
        dataset_name = self.data_args.compute_on[self.train_idx]
        # df_real = pd.read_csv(f'../Data/humor_datasets/{dataset_name}/{self.data_args.split_type}/test.csv')
        df_real = pd.read_csv(self.data_args.test_path_template.format(
            dataset=dataset_name, split_type=self.data_args.split_type, split_name='test'
        ))
        prediction_file = os.path.join(
            self.training_args.output_dir, 'predictions',
            "{dataset}_preds.csv".format(dataset=dataset_name))
        df_pred = pd.read_csv(prediction_file)
        total_count, legal_count = len(df_pred), len(df_pred)
        df_real = df_real.iloc[:total_count]

        if len(df_pred[df_pred.label == -1]) > 0:
            illegal_indices = df_pred[df_pred.label == -1].index
            print(f'there are {len(illegal_indices)} illegal indices in {dataset_name} predictions on itself')
            df_pred = df_pred.drop(labels=illegal_indices, axis=0)
            df_real = df_real.drop(labels=illegal_indices, axis=0)
            legal_count = len(df_pred)

        percent_legal = (legal_count / total_count) * 100
        self.results[ep, bs, lr, seed] = accuracy_score(df_real.label, df_pred.label), percent_legal

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
                f.write(f'accuracy = {v[0]} on {v[1]}% legal \n')

    @staticmethod
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels


if __name__ == '__main__':
    t5_trainer = T5_Trainer()
    t5_trainer.pipeline()
