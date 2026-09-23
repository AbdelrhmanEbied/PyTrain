from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EvalResult:
    metrics: dict[str, float]
    predictions: np.ndarray
    y_true: np.ndarray
    eval_time: float


@dataclass
class TrainResult:
    model: Any
    metrics: dict[str, float]
    predictions: np.ndarray
    y_true: np.ndarray
    train_time: float
    model_info: dict[str, Any]
    params: dict[str, Any]
    eval_result: EvalResult | None = None
    best_params: dict[str, Any] | None = None
    cv_results: dict[str, Any] | None = None
