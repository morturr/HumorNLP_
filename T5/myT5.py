import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List
import wandb

import datasets
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset
from filelock import FileLock
from datetime import datetime

import transformers
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
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

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


@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None)
    trained_on: str = field(default='igg')
    text_column: Optional[str] = field(default=None)
    target_column: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)
    datasets_to_predict: Optional[List[str]] = field(default=None)
    max_source_length: Optional[int] = field(default=512)
    max_target_length: Optional[int] = field(default=10)
    val_max_target_length: Optional[int] = field(default=10)
    pad_to_max_length: bool = field(default=False)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_predict_samples: Optional[int] = field(default=None)
    ignore_pad_token_for_loss: bool = field(default=True)
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        if data_args.datasets_to_predict is not None:
            if training_args.do_eval:
                path_to_predict = '../Data/humor_datasets/{dataset}/with_val/test.csv'
            else:
                path_to_predict = '../Data/humor_datasets/{dataset}/no_val/test.csv'
            for dataset in data_args.datasets_to_predict:
                curr_path = path_to_predict.format(dataset=dataset)
                print(f'dataset={dataset}, curr_path={curr_path}')
                data_files[dataset] = curr_path
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        # use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # column names for input/target
    dataset_columns = ('sentence', 'target')
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else ''
    else:
        text_column = data_args.text_column

    if data_args.target_column is None:
        target_column = dataset_columns[1] if dataset_columns is not None else ''
    else:
        target_column = data_args.target_column

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[target_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[target_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                # num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,  ## all columns??
                # load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                # num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                # load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = raw_datasets["test"]
        if data_args.datasets_to_predict:
            predict_datasets = []
            for dataset in data_args.datasets_to_predict:
                predict_datasets.append(raw_datasets[dataset])
        else:
            predict_datasets = [predict_dataset]

        for i in range(len(predict_datasets)):
            if data_args.max_predict_samples is not None:
                max_predict_samples = min(len(predict_datasets[i]), data_args.max_predict_samples)
                predict_datasets[i] = predict_datasets[i].select(range(max_predict_samples))
            with training_args.main_process_first(desc="prediction dataset map pre-processing"):
                predict_datasets[i] = predict_datasets[i].map(
                    preprocess_function,
                    batched=True,
                    # num_proc=data_args.preprocessing_num_workers,
                    remove_columns=predict_datasets[i].column_names,
                    # load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )

    # Initialize run name for wandb
    EPOCHS = [3]
    BATCH_SIZES = [8] # add 8 later again
    LRS = [5e-5]
    # LRS = [5e-5, 1e-6]
    SEEDS = [42]
    general_output_dir = training_args.output_dir

    for ep in EPOCHS:
        for bs in BATCH_SIZES:
            for lr in LRS:
                for seed in SEEDS:
                    training_args.num_train_epochs = ep
                    training_args.per_device_train_batch_size = bs
                    # training_args.per_device_eval_batch_size = bs
                    training_args.learning_rate = lr
                    training_args.seed = seed
                    time = datetime.now()
                    training_args.run_name = '{model_name}_on_{trained_on}_seed={seed}_ep={ep}_bs={bs}' \
                                             '_lr={lr}_{date}_{hour}_{minute}'.format(model_name=model_args.model_name_or_path,
                                                                        date=time.date(), hour=time.hour, minute=time.minute,
                                                                        trained_on=data_args.trained_on, seed=training_args.seed,
                                                                        ep=training_args.num_train_epochs,
                                                                        bs=training_args.per_device_train_batch_size,
                                                                        lr=training_args.learning_rate)

                    wandb.init(project='HumorNLP', name=training_args.run_name)
                    wandb.run.name = training_args.run_name
                    training_args.output_dir = '{0}/{1}'.format(general_output_dir, training_args.run_name)

                    # Initialize our Trainer
                    trainer = Seq2SeqTrainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset if training_args.do_train else None,
                        eval_dataset=eval_dataset if training_args.do_eval else None,
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
                    )

                    # Training 
                    if training_args.do_train:
                        checkpoint = None
                        if training_args.resume_from_checkpoint is not None:
                            checkpoint = training_args.resume_from_checkpoint
                        # elif last_checkpoint is not None:
                        #     checkpoint = last_checkpoint
                        train_result = trainer.train(resume_from_checkpoint=checkpoint)
                        trainer.save_model()  # Saves the tokenizer too for easy upload

                        metrics = train_result.metrics
                        max_train_samples = (
                            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
                        )
                        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

                        trainer.log_metrics("train", metrics)
                        trainer.save_metrics("train", metrics)
                        trainer.save_state()

                    # Evaluation
                    results = {}
                    if training_args.do_eval:
                        logger.info("*** Evaluate ***")
                        if isinstance(eval_dataset, dict):
                            metrics = {}
                            for eval_ds_name, eval_ds in eval_dataset.items():
                                dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
                                metrics.update(dataset_metrics)
                        else:
                            metrics = trainer.evaluate(metric_key_prefix="eval")
                        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
                            eval_dataset)
                        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

                        trainer.log_metrics("eval", metrics)
                        trainer.save_metrics("eval", metrics)

                    # Predict
                    if training_args.do_predict:
                        logger.info("*** Predict ***")
                        for i in range(len(predict_datasets)):
                            predict_results = trainer.predict(predict_datasets[i], metric_key_prefix="predict")
                            metrics = predict_results.metrics
                            max_predict_samples = (
                                data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
                            )
                            metrics["predict_samples"] = min(max_predict_samples, len(predict_datasets[i]))

                            trainer.log_metrics("predict", metrics)
                            trainer.save_metrics("predict", metrics)

                            if trainer.is_world_process_zero():
                                if training_args.predict_with_generate:
                                    predictions = predict_results.predictions
                                    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                                    predictions = tokenizer.batch_decode(
                                        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                                    )
                                    predictions = [pred.strip() for pred in predictions]
                                    output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                                    with open(output_prediction_file, "w") as writer:
                                        writer.write("\n".join(predictions))

                                else:
                                    all_tokens = predict_results.predictions[0]
                                    predicted_tokens = [[np.argmax(x) for x in tokens] for tokens in all_tokens]
                                    predicted_tokens = [[token for token in tokens if token not in tokenizer.all_special_ids]
                                                        for tokens in predicted_tokens]
                                    predictions = [tokenizer.decode(tokens) for tokens in predicted_tokens]
                                    os.makedirs(f'{training_args.output_dir}/predictions', exist_ok=True)
                                    output_prediction_file = os.path.join(
                                        training_args.output_dir, 'predictions',
                                        "{dataset}_generated_predictions.txt".format(dataset=data_args.datasets_to_predict[i]))
                                    with open(output_prediction_file, "w") as writer:
                                        writer.write("\n".join(predictions))

                    wandb.finish()

    return results


if __name__ == "__main__":
    main()
