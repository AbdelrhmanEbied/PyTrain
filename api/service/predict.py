from __future__ import annotations

import json
from typing import Any

import pandas as pd

from api.service.datasets import _json_scalar, dataset_label, find_dataset, load_frame
from api.service.preprocess import apply_operations, prepare_xy
from api.service.runs import _mlflow_client
from data.dataloader import Dataset
from models.base import Task


def _load_run_frame(
    params: dict[str, Any], target: str, task: Task
) -> tuple[pd.DataFrame, pd.Series, str]:
    source = params.get("source")
    identifier = params.get("identifier")
    split = params.get("split") or None
    dataset = params.get("dataset")
    if not identifier and dataset:
        entry = find_dataset(dataset)
        if entry:
            source = entry["source"]
            identifier = entry["identifier"]
            split = entry.get("split")
    if not source or not identifier:
        raise ValueError("Cannot resolve dataset for this run.")
    ref = Dataset(source=source, identifier=identifier, split=split)
    df = load_frame(ref)
    ops_raw = params.get("preprocess_ops") or params.get("operations")
    if ops_raw:
        try:
            ops = json.loads(ops_raw) if isinstance(ops_raw, str) else ops_raw
        except json.JSONDecodeError:
            ops = []
        if isinstance(ops, list) and ops:
            df, _ = apply_operations(df, ops)
            df = df.get_data()
    label = dataset_label(ref)
    X, y = prepare_xy(df, target, task, label)
    return X, y, label


def predict_schema(run_id: str, tracking_uri: str | None = None) -> dict[str, Any]:
    import mlflow.pyfunc

    client = _mlflow_client(tracking_uri)
    run = client.get_run(run_id)
    params = dict(run.data.params)
    task: Task = params.get("task", "classification")  # type: ignore[assignment]
    if task not in ("classification", "regression"):
        task = "classification"
    target = params.get("target") or "target"
    X, y, _ = _load_run_frame(params, target, task)

    classes: list[Any] | None = None
    if task == "classification":
        classes = [_json_scalar(v) for v in sorted(set(y.tolist()))]

    features: list[dict[str, Any]] = []
    for name in X.columns:
        col = X[name]
        sample = col.iloc[0]
        entry: dict[str, Any] = {
            "name": str(name),
            "dtype": str(col.dtype),
            "sample": _json_scalar(sample),
            "mean": float(col.mean()) if pd.api.types.is_numeric_dtype(col) else None,
            "min": float(col.min()) if pd.api.types.is_numeric_dtype(col) else None,
            "max": float(col.max()) if pd.api.types.is_numeric_dtype(col) else None,
        }
        features.append(entry)

    artifact_uri = getattr(run.info, "artifact_uri", None)
    model_name = params.get("model_name")
    loaded = False
    load_errors: list[str] = []
    for loader in (
        lambda: mlflow.sklearn.load_model(f"runs:/{run_id}/model"),
        lambda: mlflow.pyfunc.load_model(f"runs:/{run_id}/model"),
    ):
        try:
            loader()
            loaded = True
            break
        except Exception as exc:
            load_errors.append(str(exc))
    if not loaded:
        raise FileNotFoundError(
            f"No model artifact found for run {run_id}: {load_errors[-1] if load_errors else 'unknown'}"
        )

    return {
        "run_id": run_id,
        "task": task,
        "target": params.get("target"),
        "model_name": model_name,
        "features": features,
        "n_features": len(features),
        "classes": classes,
        "artifact_uri": artifact_uri,
    }


def predict_run(req: Any) -> dict[str, Any]:
    import mlflow.pyfunc
    import numpy as np

    client = _mlflow_client(req.tracking_uri)
    run = client.get_run(req.run_id)
    params = dict(run.data.params)
    task: Task = params.get("task", "classification")  # type: ignore[assignment]
    if task not in ("classification", "regression"):
        task = "classification"
    target = params.get("target") or "target"
    X, _, _ = _load_run_frame(params, target, task)

    missing = [c for c in X.columns if c not in req.features]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    row = {c: req.features[c] for c in X.columns}
    frame = pd.DataFrame([row], columns=X.columns)
    for col in X.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.astype(X.dtypes.to_dict())

    model = None
    errors: list[str] = []
    for loader in (
        lambda: mlflow.sklearn.load_model(f"runs:/{req.run_id}/model"),
        lambda: mlflow.pyfunc.load_model(f"runs:/{req.run_id}/model"),
    ):
        try:
            model = loader()
            break
        except Exception as exc:
            errors.append(str(exc))
    if model is None:
        raise FileNotFoundError(f"No model artifact found for run {req.run_id}: {errors[-1]}")

    use_frame = getattr(model, "feature_names_in_", None) is not None
    x_in = frame if use_frame else frame.to_numpy()
    prediction = np.asarray(model.predict(x_in))
    pred = _json_scalar(prediction[0])

    probabilities: dict[str, float] | None = None
    classes: list[Any] | None = None
    if task == "classification":
        classes = [_json_scalar(v) for v in getattr(model, "classes_", [])]
        if hasattr(model, "predict_proba"):
            try:
                proba = np.asarray(model.predict_proba(x_in))[0]
                if classes:
                    probabilities = {str(c): float(p) for c, p in zip(classes, proba, strict=False)}
                else:
                    probabilities = {str(i): float(p) for i, p in enumerate(proba)}
            except Exception:
                probabilities = None

    return {
        "run_id": req.run_id,
        "task": task,
        "prediction": pred,
        "probabilities": probabilities,
        "classes": classes,
    }
