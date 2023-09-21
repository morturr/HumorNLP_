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
from Utils.utils import print_cur_time
from Model.HumorTrainer import HumorTrainer

logger = logging.getLogger(__name__)
# wandb.init(project='HumorNLP')

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


class T5_Trainer(HumorTrainer):
    def __init__(self):
        self.padding, self.max_target_length, self.prefix = None, None, None
        self.data_collator = None
        self.target_column = None

        HumorTrainer.__init__(self)

    def config_and_tokenizer(self):
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

        # Metric
        self.metric = evaluate.load("rouge")

    def init_model(self):
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

    def set_data_attr(self):
        self.prefix = self.data_args.source_prefix if self.data_args.source_prefix is not None else ""

        # column names for input/target
        self.dataset_columns = ('t5_sentence', 'target')

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

        HumorTrainer.set_data_attr(self)

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
        HumorTrainer.preprocess_datasets(self, remove_columns=True)

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

        HumorTrainer.train(self)

    def predict(self, ep, bs, lr, seed):
        # Predict
        if self.training_args.do_predict:
            logger.info("*** Predict ***")
            print_cur_time(f'START PREDICT OF {self.training_args.run_name}')

            for i in range(len(self.predict_datasets)):
                print_cur_time(f'START PREDICT ON {self.data_args.datasets_to_predict[i]}')
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

                        self.save_predictions(predictions, self.data_args.datasets_to_predict[i],
                                              ep, bs, lr, seed)

                print_cur_time(f'END PREDICT ON {self.data_args.datasets_to_predict[i]}')

            print_cur_time(f'END PREDICT OF {self.training_args.run_name}')

    def save_predictions(self, predictions, predict_dataset, ep, bs, lr, seed):
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

        if self.data_args.test_path_template:
            df_real_path = self.data_args.test_path_template.format(
                dataset=predict_dataset, split_type=self.data_args.split_type, split_name='test'
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
        df_pred['t5_sentence'] = df_real['t5_sentence']
        df_pred['id'] = df_real['id']
        df_pred['true_label'] = df_real['label']
        cols = ['id', 't5_sentence', 'target', 'label', 'true_label', 'original', 'edited']
        df_pred = df_pred[cols]

        # save predictions (with illegals)
        os.makedirs(f'{self.training_args.output_dir}/predictions', exist_ok=True)
        output_prediction_file = os.path.join(
            self.training_args.output_dir, 'predictions',
            "{dataset}_preds.csv".format(dataset=predict_dataset))
        df_pred.to_csv(output_prediction_file, index=False)

        if len(df_pred[df_pred.label == -1]) > 0:
            illegal_indices = df_pred[df_pred.label == -1].index
            print(f'there are {len(illegal_indices)} illegal indices in {self.data_args.trained_on[self.train_idx]}'
                  f' predictions on {predict_dataset}')
            df_pred = df_pred.drop(labels=illegal_indices, axis=0)
            df_real = df_real.drop(labels=illegal_indices, axis=0)

        accuracy = float("%.4f" % accuracy_score(df_real.label, df_pred.label))
        recall = float("%.4f" % recall_score(df_real.label, df_pred.label))
        precision = float("%.4f" % precision_score(df_real.label, df_pred.label))

        row_to_final_results = {'model': self.model_args.model_name_or_path,
                                'trained_on': self.data_args.trained_on[self.train_idx],
                                'epoch': ep, 'batch_size': bs,
                                'learning_rate': lr, 'seed': seed,
                                'predict_on': predict_dataset,
                                'accuracy': accuracy, 'recall': recall, 'precision': precision}

        print(row_to_final_results)

        self.final_results_df = self.final_results_df.append([row_to_final_results])

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
