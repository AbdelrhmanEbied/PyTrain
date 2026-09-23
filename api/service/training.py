from __future__ import annotations

import json
from typing import Any

import pandas as pd

from api.schemas import TrainRequest
from api.service.datasets import (
    dataset_label,
    default_target,
    load_frame,
    resolve_ref,
)
from api.service.models import make_model, make_param_config, make_trainer
from api.service.preprocess import apply_operations, prepare_xy
from data.preprocessing import Preprocessor, PreprocessorConfig
from models.base import Task
from training.results import EvalResult, TrainResult


def split_arrays(
    X: pd.DataFrame,
    y: pd.Series,
    target: str,
    task: Task,
    test_size: float,
    validation_size: float | None,
    stratify: bool,
    random_state: int,
) -> dict[str, Any]:
    frame = X.copy()
    frame[target] = y.values
    pp = Preprocessor(PreprocessorConfig(df=frame))
    split = pp.split(
        target,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
        stratify=stratify,
    )

    def to_xy(X_part: pd.DataFrame | None, y_part: pd.Series | None):
        if X_part is None or y_part is None:
            return None, None
        return X_part.values, y_part.values

    X_train, y_train = to_xy(split.X_train, split.y_train)
    X_test, y_test = to_xy(split.X_test, split.y_test)
    X_val, y_val = to_xy(split.X_validation, split.y_validation)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def train_result_to_dict(
    result: TrainResult,
    test_result: EvalResult,
    n_train: int,
    n_val: int,
    n_test: int,
    n_features: int,
) -> dict[str, Any]:
    return {
        "model_info": result.model_info,
        "params": result.params,
        "best_params": result.best_params,
        "cv_results": result.cv_results,
        "train_metrics": result.metrics,
        "val_metrics": result.eval_result.metrics if result.eval_result else None,
        "test_metrics": test_result.metrics,
        "train_time": result.train_time,
        "eval_time": test_result.eval_time,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_features": n_features,
    }


def train_pipeline_snapshot(
    payload: dict[str, Any],
    label: str,
    target: str,
    task: str,
    df_shape: tuple[int, int],
    logged: bool,
    preprocess_steps: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [
        {
            "operation": "load_dataset",
            "label": "Load data",
            "args": {"source": label, "rows": df_shape[0], "cols": df_shape[1]},
            "status": "done",
        },
    ]
    for i, step in enumerate(preprocess_steps or []):
        steps.append(
            {
                "operation": step["operation"],
                "label": step["operation"],
                "args": step.get("args", {}),
                "rows": step.get("rows"),
                "cols": step.get("cols"),
                "status": "done",
                "idx": i + 1,
            }
        )
    steps.extend(
        [
            {
                "operation": "prepare_features",
                "label": "Prepare features",
                "args": {"target": target, "n_features": payload["n_features"]},
                "status": "done",
            },
            {
                "operation": "split_data",
                "label": "Split data",
                "args": {
                    "train": payload["n_train"],
                    "val": payload["n_val"],
                    "test": payload["n_test"],
                },
                "status": "done",
            },
            {
                "operation": "train_model",
                "label": "Train model",
                "args": {
                    "model_name": payload["model_info"]["model_name"],
                    "task": task,
                    "framework": payload["model_info"]["framework"],
                },
                "status": "done",
            },
            {
                "operation": "evaluate",
                "label": "Evaluate",
                "args": {"n_test": payload["n_test"]},
                "status": "done",
            },
        ]
    )
    if logged:
        steps.append(
            {
                "operation": "log_mlflow",
                "label": "Log to MLflow",
                "args": {"run_id": payload.get("mlflow_run_id") or ""},
                "status": "done",
            }
        )
    steps.append({"operation": "finish", "label": "Complete", "args": {}, "status": "done"})
    return steps


def run_pipeline_snapshot(params: dict[str, Any]) -> list[dict[str, Any]]:
    identifier = params.get("identifier") or params.get("dataset") or "dataset"
    source = params.get("source") or ""
    target = params.get("target") or ""
    model_name = params.get("model_name") or ""
    steps = [
        {
            "operation": "load_dataset",
            "label": "Load data",
            "args": {"source": source, "identifier": identifier},
            "status": "done",
        },
        {
            "operation": "prepare_features",
            "label": "Prepare features",
            "args": {"target": target},
            "status": "done",
        },
        {
            "operation": "split_data",
            "label": "Split data",
            "args": {
                "test_size": params.get("test_size", ""),
                "validation_size": params.get("validation_size", ""),
            },
            "status": "done",
        },
        {
            "operation": "train_model",
            "label": "Train model",
            "args": {"model_name": model_name, "task": params.get("task", "")},
            "status": "done",
        },
        {
            "operation": "evaluate",
            "label": "Metrics",
            "args": {"status": "logged"},
            "status": "done",
        },
    ]
    if params.get("logged_to_mlflow") or params.get("mlflow_run_id"):
        steps.append(
            {
                "operation": "log_mlflow",
                "label": "MLflow",
                "args": {"run_id": str(params.get("mlflow_run_id") or "")[:8]},
                "status": "done",
            }
        )
    return steps


def run_training(req: TrainRequest, on_stage: Any = None) -> dict[str, Any]:
    from api.service.runs import DEFAULT_TRACKING_URI

    def stage(name: str, detail: str | None = None) -> None:
        if on_stage is not None:
            on_stage(name, detail)

    ref = resolve_ref(req.source, req.identifier, req.split, req.dataset)
    label = dataset_label(ref)
    target = req.target or default_target(label if req.dataset is None else req.dataset)
    stage("loading", f"{label}")

    df = load_frame(ref)
    preprocess_steps: list[dict[str, Any]] = []
    preprocess_ops: list[dict[str, Any]] = []
    if req.operations:
        stage("preprocessing", f"{len(req.operations)} ops")
        df, preprocess_steps = apply_operations(df, req.operations)
        preprocess_ops = df.get_operations()
        df = df.get_data()
    X, y = prepare_xy(df, target, req.task, label)
    stage("preparing", f"{X.shape[1]} features")
    splits = split_arrays(
        X,
        y,
        target,
        req.task,
        test_size=req.test_size,
        validation_size=req.validation_size,
        stratify=req.stratify and req.task == "classification",
        random_state=req.random_state,
    )
    model = make_model(req.model_name, req.task, req.params)
    trainer = make_trainer(model, req.task)
    param_config = make_param_config(req.param_config.model_dump()) if req.param_config else None
    mode = "grid search" if param_config is not None else "fit"
    stage("training", f"{req.model_name} · {mode}")
    result = trainer.train(
        splits["X_train"],
        splits["y_train"],
        X_val=splits["X_val"],
        y_val=splits["y_val"],
        param_config=param_config,
    )
    stage("evaluating", f"{len(splits['y_test'])} test samples")
    test_result = trainer.evaluate(splits["X_test"], splits["y_test"])

    df_shape = df.shape
    payload = train_result_to_dict(
        result,
        test_result,
        n_train=len(splits["y_train"]),
        n_val=len(splits["y_val"]) if splits["y_val"] is not None else 0,
        n_test=len(splits["y_test"]),
        n_features=splits["X_train"].shape[1],
    )
    payload["logged_to_mlflow"] = False
    payload["mlflow_run_id"] = None
    payload["preprocess_steps"] = preprocess_steps
    payload["preprocess_ops"] = preprocess_ops

    if req.log_mlflow:
        stage("logging", req.experiment_name)
        try:
            from mlflow_integration import MLflowTracker

            tracker = MLflowTracker(
                experiment_name=req.experiment_name,
                tracking_uri=req.tracking_uri or DEFAULT_TRACKING_URI,
            )
            with tracker:
                tracker.log_params(
                    {
                        "dataset": req.dataset or label,
                        "source": ref.source,
                        "identifier": str(ref.identifier),
                        "split": ref.split or "",
                        "target": target,
                        "test_size": req.test_size,
                        "validation_size": str(req.validation_size),
                        "stratify": req.stratify,
                        "random_state": req.random_state,
                        "task": req.task,
                        "model_name": req.model_name,
                        "n_preprocess_ops": str(len(req.operations)),
                        "preprocess_ops": json.dumps(req.operations),
                        "preprocess_steps": json.dumps(preprocess_steps),
                    }
                )
                if result.best_params is not None:
                    tracker.log_params({"best_params": json.dumps(result.best_params)})
                tracker.log_train_result(result)
                tracker.log_metrics({f"test_{k}": v for k, v in test_result.metrics.items()})
                run_id = tracker.active_run_id
        except Exception as exc:
            raise ValueError(f"MLflow logging failed: {exc}") from exc
        payload["logged_to_mlflow"] = True
        payload["mlflow_run_id"] = run_id

    payload["pipeline"] = train_pipeline_snapshot(
        payload,
        label=label,
        target=target,
        task=req.task,
        df_shape=df_shape,
        logged=payload["logged_to_mlflow"],
        preprocess_steps=preprocess_steps,
    )
    return payload
