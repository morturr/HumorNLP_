
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class FlanTrainingArguments:
    model_name: Optional[str] = field(default='google/flan-t5-base')
    dataset_name: Optional[str] = field(default=None)
    epochs: Optional[List[int]] = field(default_factory=lambda: [2, 3])
    batch_sizes: Optional[List[int]] = field(default_factory=lambda: [4, 8, 16])
    learning_rates: Optional[List[float]] = field(default_factory=lambda: [5e-5, 1e-5, 1e-6])
    seeds: Optional[List[int]] = field(default_factory=lambda: [5, 19, 28, 42])
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_predict_samples: Optional[int] = field(default=None)