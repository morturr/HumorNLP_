import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None)
    trained_on: str = field(default='igg')
    split_type: Optional[str] = field(default='with_val_fixed_train')
    text_column: Optional[str] = field(default=None)
    target_column: Optional[str] = field(default=None)
    label_column: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)
    datasets_to_predict: Optional[List[str]] = field(default=None)
    epochs: Optional[List[int]] = field(default_factory=lambda: [3])
    batch_sizes: Optional[List[int]] = field(default_factory=lambda: [8])
    learning_rates: Optional[List[float]] = field(default_factory=lambda: [1e-5])
    seeds: Optional[List[int]] = field(default_factory=lambda: [42])
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
    save_model: Optional[bool] = field(default=True)
    save_metrics: Optional[bool] = field(default=True)
    save_state: Optional[bool] = field(default=True)


def print_str(s):
    print(f'\n ~~ {s} ~~ \n')


def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_train_samples', type=int)
    parser.add_argument('n_test_samples', type=int)
    parser.add_argument('task', type=str)  # 'hyperparams', 'train'
    return parser.parse_args()


def print_cur_time(status):
    """ this function print the current time to the stdout """
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(status + " Current Time = " + current_time)
