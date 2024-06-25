
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class FlanTrainingArguments:
    model_name: Optional[str] = field(default='google/flan-t5-base')
    dataset_name: Optional[str] = field(default=None)
    leave_one_out_datasets: Optional[List[str]] = field(default=None)
    epochs: Optional[List[int]] = field(default_factory=lambda: [2, 3])
    batch_sizes: Optional[List[int]] = field(default_factory=lambda: [4, 8, 16])
    learning_rates: Optional[List[float]] = field(default_factory=lambda: [5e-5, 1e-5, 1e-6])
    seeds: Optional[List[int]] = field(default_factory=lambda: [5, 19, 28, 42])
    samples_percent: Optional[int] = field(default=None)


@dataclass
class FlanEvaluationArguments:
    model_id: Optional[str] = field(default=None)
    datasets: Optional[List[str]] = field(default_factory=lambda: ['amazon', 'dadjokes', 'headlines',
                                                                   'one_liners', 'yelp_reviews'])
    samples_percent: Optional[int] = field(default=None)
