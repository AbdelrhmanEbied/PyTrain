from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow

from training.results import TrainResult

SKOPS_TRUSTED_TYPES = [
    "xgboost.sklearn.XGBClassifier",
    "xgboost.sklearn.XGBRegressor",
    "xgboost.sklearn.XGBModel",
    "xgboost.sklearn.XGBRFClassifier",
    "xgboost.sklearn.XGBRFRegressor",
    "xgboost.core.Booster",
    "numpy.dtype",
    "numpy.ndarray",
    "tuple",
]


class MLflowTracker:
    def __init__(
        self,
        experiment_name: str = "pytrain",
        tracking_uri: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @property
    def ui_url(self) -> str:
        return mlflow.get_tracking_uri().replace("sqlite:///", "http://localhost:5000/#")

    @property
    def active_run_id(self) -> str | None:
        run = mlflow.active_run()
        if run is None:
            return None
        return run.info.run_id

    def start_run(self, run_name: str | None = None) -> Any:
        return mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        mlflow.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        mlflow.sklearn.log_model(
            model,
            name=artifact_path,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    def log_train_result(self, result: TrainResult) -> None:
        self.log_params(result.params)
        self.log_params(result.model_info)
        self.log_metrics(result.metrics)
        self.log_metrics({"train_time": result.train_time})
        if result.best_params is not None:
            self.log_params(result.best_params)
        if result.cv_results is not None:
            self.log_metrics(
                {f"cv_{k}": v for k, v in result.cv_results.items() if isinstance(v, (int, float))}
            )
        if result.eval_result is not None:
            eval_metrics = {f"val_{k}": v for k, v in result.eval_result.metrics.items()}
            self.log_metrics(eval_metrics)
        self.log_model(result.model)

    def __enter__(self) -> MLflowTracker:
        mlflow.start_run()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        mlflow.end_run()
