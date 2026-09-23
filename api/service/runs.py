from __future__ import annotations

import json
import os
from typing import Any

from api.service.datasets import (
    _json_scalar,
    dataset_label,
    find_dataset,
    load_frame,
)
from api.service.preprocess import prepare_xy
from api.service.training import run_pipeline_snapshot, split_arrays
from data.dataloader import Dataset
from models.base import Task

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def mlflow_status(
    tracking_uri: str | None = None,
    experiment_name: str = "pytrain",
    ui_url: str | None = None,
) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    ui = (
        ui_url
        or os.environ.get("MLFLOW_UI_URL")
        or os.environ.get("MLFLOW_TRACKING_UI")
        or "http://127.0.0.1:5000"
    ).rstrip("/")
    server_up = False
    try:
        req = urllib.request.Request(ui, method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as res:
            server_up = 200 <= getattr(res, "status", 200) < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        server_up = False

    run_count = 0
    try:
        client = _mlflow_client(uri)
        experiments = client.search_experiments()
        ids = [e.experiment_id for e in experiments]
        if ids:
            runs = client.search_runs(experiment_ids=ids, max_results=1)
            run_count = len(runs) if runs is not None else 0
            if runs:
                run_count = int(runs[0].info.run_id is not None)
        total = 0
        if ids:
            page = client.search_runs(experiment_ids=ids, max_results=1000)
            total = len(page)
        run_count = total
    except Exception:
        run_count = 0

    return {
        "ui_url": ui,
        "tracking_uri": uri,
        "server_up": server_up,
        "experiment_name": experiment_name,
        "run_count": run_count,
    }


def _mlflow_client(tracking_uri: str | None):
    import mlflow

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    from mlflow.tracking import MlflowClient

    return MlflowClient()


def _run_to_dict(run: Any) -> dict[str, Any]:
    params = dict(run.data.params)
    return {
        "run_id": run.info.run_id,
        "name": run.data.tags.get("mlflow.runName"),
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "params": params,
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags),
        "pipeline": run_pipeline_snapshot(params),
    }


def list_runs(tracking_uri: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    client = _mlflow_client(tracking_uri)
    experiments = client.search_experiments()
    if not experiments:
        return []
    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in experiments],
        max_results=limit,
        order_by=["attributes.start_time DESC"],
    )
    return [_run_to_dict(run) for run in runs]


def get_run(run_id: str, tracking_uri: str | None = None) -> dict[str, Any]:
    client = _mlflow_client(tracking_uri)
    run = client.get_run(run_id)
    return _run_to_dict(run)


def get_run_details(run_id: str, tracking_uri: str | None = None) -> dict[str, Any]:
    client = _mlflow_client(tracking_uri)
    run = client.get_run(run_id)
    info = run.info
    data = run.data
    params = dict(data.params)
    base = _run_to_dict(run)

    duration_ms = None
    if info.start_time and info.end_time:
        duration_ms = int(info.end_time - info.start_time)

    preprocess_ops: list[dict[str, Any]] = []
    preprocess_steps: list[dict[str, Any]] = []
    for key in ("preprocess_ops", "operations", "preprocess_operations"):
        raw = params.get(key)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            preprocess_ops = parsed
            break

    for key in ("preprocess_steps",):
        raw = params.get(key)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            preprocess_steps = parsed
            break

    model_hyperparams = {}
    for key, value in params.items():
        if key.startswith("estimator__"):
            model_hyperparams[key.removeprefix("estimator__")] = value
    if not model_hyperparams:
        skip = {
            "dataset",
            "source",
            "identifier",
            "split",
            "target",
            "test_size",
            "validation_size",
            "stratify",
            "random_state",
            "task",
            "model_name",
            "n_preprocess_ops",
            "preprocess_ops",
            "preprocess_steps",
            "operations",
            "framework",
            "experiment",
            "mlflow.runName",
        }
        model_hyperparams = {k: v for k, v in params.items() if k not in skip}

    grid_config = None
    if params.get("grid_cv") or params.get("best_params"):
        grid_config = {
            "cv": params.get("grid_cv") or params.get("cv"),
            "scoring": params.get("grid_scoring") or params.get("scoring"),
            "best_params": (
                json.loads(params["best_params"])
                if params.get("best_params", "").strip().startswith("{")
                else params.get("best_params")
            ),
        }

    dataset_info = {
        "dataset": params.get("dataset"),
        "source": params.get("source"),
        "identifier": params.get("identifier"),
        "split": params.get("split"),
        "target": params.get("target"),
        "n_preprocess_ops": params.get("n_preprocess_ops"),
    }

    split_config = {
        "test_size": params.get("test_size"),
        "validation_size": params.get("validation_size"),
        "stratify": params.get("stratify"),
        "random_state": params.get("random_state"),
        "task": params.get("task"),
        "model_name": params.get("model_name"),
    }

    artifacts: list[dict[str, Any]] = []
    try:
        for info_art in client.list_artifacts(run_id):
            artifacts.append(
                {
                    "path": info_art.path,
                    "is_dir": bool(info_art.is_dir),
                    "file_size": getattr(info_art, "file_size", None),
                }
            )
            if info_art.is_dir:
                for child in client.list_artifacts(run_id, info_art.path):
                    artifacts.append(
                        {
                            "path": child.path,
                            "is_dir": bool(child.is_dir),
                            "file_size": getattr(child, "file_size", None),
                        }
                    )
    except Exception:
        artifacts = []

    metric_history: dict[str, list[dict[str, Any]]] = {}
    for key in list(data.metrics.keys()):
        try:
            history = client.get_metric_history(run_id, key)
            metric_history[key] = [
                {"value": m.value, "timestamp": m.timestamp, "step": m.step} for m in history
            ]
        except Exception:
            metric_history[key] = [
                {"value": data.metrics[key], "timestamp": info.start_time, "step": 0}
            ]

    return {
        **base,
        "duration_ms": duration_ms,
        "lifecycle_stage": getattr(info, "lifecycle_stage", None),
        "artifact_uri": getattr(info, "artifact_uri", None),
        "user_id": data.tags.get("mlflow.user"),
        "preprocess_ops": preprocess_ops,
        "preprocess_steps": preprocess_steps,
        "split_config": split_config,
        "model_hyperparams": model_hyperparams,
        "grid_config": grid_config,
        "dataset_info": dataset_info,
        "artifacts": artifacts,
        "metric_history": metric_history,
    }


def rename_run(run_id: str, name: str, tracking_uri: str | None = None) -> dict[str, Any]:
    client = _mlflow_client(tracking_uri)
    client.update_run(run_id, name=name)
    return get_run_details(run_id, tracking_uri)


def delete_run(run_id: str, tracking_uri: str | None = None) -> dict[str, Any]:
    client = _mlflow_client(tracking_uri)
    client.delete_run(run_id)
    return {"run_id": run_id, "deleted": True}


def restore_run(run_id: str, tracking_uri: str | None = None) -> dict[str, Any]:
    client = _mlflow_client(tracking_uri)
    client.restore_run(run_id)
    return {"run_id": run_id, "restored": True}


def evaluate_run(req: Any) -> dict[str, Any]:
    import time

    import mlflow.pyfunc
    import numpy as np

    from training.metrics import compute_metrics

    client = _mlflow_client(req.tracking_uri)
    run = client.get_run(req.run_id)
    params = dict(run.data.params)
    task: Task = params.get("task", "classification")  # type: ignore[assignment]
    if task not in ("classification", "regression"):
        task = "classification"

    source = req.source or params.get("source")
    identifier = req.identifier or params.get("identifier")
    split = req.split if req.split is not None else (params.get("split") or None)
    dataset = req.dataset or params.get("dataset")
    if not identifier and dataset:
        entry = find_dataset(dataset)
        if entry:
            source = entry["source"]
            identifier = entry["identifier"]
            split = entry.get("split")
    if not source or not identifier:
        raise ValueError("Cannot resolve dataset for evaluation.")

    target = req.target or params.get("target") or "target"
    ref = Dataset(source=source, identifier=identifier, split=split)
    df = load_frame(ref)
    X, y = prepare_xy(df, target, task, dataset_label(ref))

    random_state = int(params.get("random_state", req.random_state))
    test_size = float(params.get("test_size", req.test_size))
    raw_val = params.get("validation_size", str(req.validation_size))
    validation_size = None if raw_val in ("None", "none", "") else float(raw_val)
    splits = split_arrays(
        X,
        y,
        target,
        task,
        test_size=test_size,
        validation_size=validation_size,
        stratify=req.stratify and task == "classification",
        random_state=random_state,
    )

    model = None
    load_errors = []
    for loader in (
        lambda: mlflow.pyfunc.load_model(f"runs:/{req.run_id}/model"),
        lambda: mlflow.sklearn.load_model(f"runs:/{req.run_id}/model"),
    ):
        try:
            model = loader()
            break
        except Exception as exc:
            load_errors.append(str(exc))
    if model is None:
        raise FileNotFoundError(f"No model artifact found for run {req.run_id}: {load_errors[-1]}")

    start = time.perf_counter()
    predictions = np.asarray(model.predict(splits["X_test"]))
    eval_time = time.perf_counter() - start
    y_true = np.asarray(splits["y_test"])
    metrics = compute_metrics(task, y_true, predictions)

    confusion: dict[str, Any] | None = None
    samples: list[dict[str, float]] = []
    if task == "classification":
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix

        labels = sorted(set(y_true.tolist()) | set(predictions.tolist()))
        matrix = sk_confusion_matrix(y_true, predictions, labels=labels)
        confusion = {
            "labels": [_json_scalar(v) for v in labels],
            "matrix": matrix.tolist(),
        }
    max_points = 300
    n = len(y_true)
    if n > max_points:
        step = max(n // max_points, 1)
        indices = range(0, n, step)
    else:
        indices = range(n)
    for i in indices:
        samples.append(
            {
                "actual": float(y_true[i]),
                "predicted": float(predictions[i]),
            }
        )

    return {
        "run_id": req.run_id,
        "metrics": metrics,
        "n_test": int(n),
        "eval_time": eval_time,
        "task": task,
        "confusion_matrix": confusion,
        "samples": samples,
    }
