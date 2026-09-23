from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV

from models.base import InvalidParameterError, Task
from training.base import BaseTrainer
from training.metrics import compute_metrics
from training.param_config import ParamConfig
from training.results import EvalResult, TrainResult

_DEFAULT_SCORING = {
    "classification": "accuracy",
    "regression": "r2",
}


class EstimatorTrainer(BaseTrainer):
    def __init__(self, model: Any, task: Task) -> None:
        self.model = model
        self.task = task
        self._estimator: Any = None

    @property
    def estimator(self) -> Any:
        if self._estimator is None:
            raise RuntimeError("Trainer has not been fitted. Call train() first.")
        return self._estimator

    def train(
        self,
        X: Any,
        y: Any,
        X_val: Any = None,
        y_val: Any = None,
        param_config: ParamConfig | None = None,
    ) -> TrainResult:
        best_params: dict[str, Any] | None = None
        cv_results: dict[str, Any] | None = None

        if param_config is not None:
            self._validate_grid(param_config.grid)
            start = time.perf_counter()
            self._estimator, best_params, cv_results = self._grid_search(X, y, param_config)
            train_time = time.perf_counter() - start
            self.model.set_params(**best_params)
        else:
            self._estimator = self.model.build()
            start = time.perf_counter()
            self._fit(self._estimator, X, y)
            train_time = time.perf_counter() - start

        predictions = self._predict(self._estimator, X)
        y_true = np.asarray(y)
        metrics = compute_metrics(self.task, y_true, predictions)

        eval_result = None
        if X_val is not None and y_val is not None:
            eval_result = self.evaluate(X_val, y_val)

        return TrainResult(
            model=self._estimator,
            metrics=metrics,
            predictions=predictions,
            y_true=y_true,
            train_time=train_time,
            model_info=self.model.get_model_info(),
            params=self.model.get_params(),
            eval_result=eval_result,
            best_params=best_params,
            cv_results=cv_results,
        )

    def evaluate(self, X: Any, y: Any) -> EvalResult:
        start = time.perf_counter()
        predictions = self._predict(self.estimator, X)
        eval_time = time.perf_counter() - start

        y_true = np.asarray(y)
        metrics = compute_metrics(self.task, y_true, predictions)

        return EvalResult(
            metrics=metrics,
            predictions=predictions,
            y_true=y_true,
            eval_time=eval_time,
        )

    def predict(self, X: Any) -> np.ndarray:
        return self._predict(self.estimator, X)

    def predict_proba(self, X: Any) -> np.ndarray | None:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        return None

    def _grid_search(
        self,
        X: Any,
        y: Any,
        param_config: ParamConfig,
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        estimator = self.model.build()
        scoring = param_config.scoring or _DEFAULT_SCORING[self.task]

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_config.grid,
            cv=param_config.cv,
            scoring=scoring,
            refit=param_config.refit,
            n_jobs=param_config.n_jobs,
            verbose=param_config.verbose,
        )
        search.fit(X, y)

        cv_summary = {
            "best_score": float(search.best_score_),
            "best_index": int(search.best_index_),
            "scoring": scoring,
            "cv": param_config.cv,
            "mean_test_score": float(np.max(search.cv_results_["mean_test_score"])),
            "std_test_score": float(search.cv_results_["std_test_score"][search.best_index_]),
            "mean_fit_time": float(np.mean(search.cv_results_["mean_fit_time"])),
            "n_candidates": len(search.cv_results_["params"]),
        }

        return search.best_estimator_, dict(search.best_params_), cv_summary

    def _validate_grid(self, grid: dict[str, list[Any]]) -> None:
        schema = {s.name: s for s in self.model.get_parameter_schema()}
        invalid = set(grid.keys()) - set(schema.keys())
        if invalid:
            raise InvalidParameterError(
                f"Unknown parameters for {self.model.model_name!r}: {invalid}"
            )
        for name, values in grid.items():
            for value in values:
                if value is None and not schema[name].nullable:
                    raise InvalidParameterError(f"Parameter {name!r} does not accept None")
                if value is None:
                    continue
                s = schema[name]
                if s.choices is not None and value not in s.choices:
                    raise InvalidParameterError(
                        f"Parameter {name!r} must be one of {s.choices}, got {value!r}"
                    )
                if s.min_value is not None and value < s.min_value:
                    raise InvalidParameterError(
                        f"Parameter {name!r} must be >= {s.min_value}, got {value!r}"
                    )
                if s.max_value is not None and value > s.max_value:
                    raise InvalidParameterError(
                        f"Parameter {name!r} must be <= {s.max_value}, got {value!r}"
                    )

    def _fit(self, estimator: Any, X: Any, y: Any) -> None:
        estimator.fit(X, y)

    def _predict(self, estimator: Any, X: Any) -> np.ndarray:
        return np.asarray(estimator.predict(X))
