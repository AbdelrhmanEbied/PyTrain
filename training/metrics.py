from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

CLASSIFICATION_METRICS = {
    "accuracy": accuracy_score,
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted", zero_division=0),
    "precision": lambda y_true, y_pred: precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "recall": lambda y_true, y_pred: recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
}

REGRESSION_METRICS = {
    "mse": mean_squared_error,
    "rmse": lambda y_true, y_pred: float(np.sqrt(mean_squared_error(y_true, y_pred))),
    "mae": mean_absolute_error,
    "r2": r2_score,
}


def compute_metrics(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    if task == "classification":
        results = {name: float(fn(y_true, y_pred)) for name, fn in CLASSIFICATION_METRICS.items()}
        if y_proba is not None:
            try:
                results["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                )
            except ValueError:
                pass
        return results
    if task == "regression":
        return {name: float(fn(y_true, y_pred)) for name, fn in REGRESSION_METRICS.items()}
    raise ValueError(f"Unknown task: {task!r}")
