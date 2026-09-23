from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from models.base import Task
from training.param_config import ParamConfig
from training.results import EvalResult, TrainResult


class BaseTrainer(ABC):
    task: Task

    @abstractmethod
    def train(
        self,
        X: Any,
        y: Any,
        X_val: Any = None,
        y_val: Any = None,
        param_config: ParamConfig | None = None,
    ) -> TrainResult: ...

    @abstractmethod
    def evaluate(self, X: Any, y: Any) -> EvalResult: ...

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task!r})"
