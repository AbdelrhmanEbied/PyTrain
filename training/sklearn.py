from __future__ import annotations

from typing import Any

from models.base import Task
from training.estimator import EstimatorTrainer


class SklearnTrainer(EstimatorTrainer):
    def __init__(self, model: Any, task: Task) -> None:
        super().__init__(model, task)
