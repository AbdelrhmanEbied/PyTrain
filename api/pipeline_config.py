from __future__ import annotations

import json
from typing import Any

import yaml

from api.schemas import ParamConfigIn, TrainRequest
from api.service import PREPROCESSOR_METHODS, list_models
from models.base import ModelError

ALLOWED_TOP = {
    "dataset",
    "source",
    "identifier",
    "split",
    "target",
    "training",
    "preprocessing",
    "operations",
    "model",
    "params",
    "param_config",
    "test_size",
    "validation_size",
    "stratify",
    "random_state",
    "log_mlflow",
    "experiment_name",
    "tracking_uri",
    "task",
}

DATASET_KEYS = {"source", "identifier", "split", "dataset", "target"}
TRAINING_KEYS = {
    "model_name",
    "task",
    "params",
    "param_config",
    "test_size",
    "validation_size",
    "stratify",
    "random_state",
    "log_mlflow",
    "experiment_name",
    "tracking_uri",
    "target",
    "source",
    "identifier",
    "split",
    "dataset",
    "operations",
}


def parse_config_text(text: str, fmt: str | None = None) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Config is empty.")
    detected = (fmt or "").lower()
    if not detected:
        if raw.lstrip().startswith("{"):
            detected = "json"
        else:
            detected = "yaml"
    if detected == "json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
    elif detected in {"yaml", "yml"}:
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML: {exc}") from exc
    else:
        raise ValueError(f"Unsupported format: {fmt!r}. Use json or yaml.")
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/object.")
    return data


def _validate_operations(ops: Any) -> list[dict[str, Any]]:
    if ops is None:
        return []
    if not isinstance(ops, list):
        raise ValueError("operations must be a list.")
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(ops):
        if not isinstance(raw, dict):
            raise ValueError(f"operations[{i}] must be an object.")
        op = raw.get("operation")
        if op not in PREPROCESSOR_METHODS:
            raise ValueError(
                f"operations[{i}].operation {op!r} is invalid. "
                f"Available: {sorted(PREPROCESSOR_METHODS)}"
            )
        out.append(dict(raw))
    return out


def _validate_param_config(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("param_config must be an object.")
    cfg = ParamConfigIn(**{k: v for k, v in raw.items() if k in ParamConfigIn.model_fields})
    if not cfg.grid:
        raise ValueError("param_config.grid must be a non-empty object.")
    return cfg.model_dump()


def _validate_model(model_name: str, task: str, params: dict[str, Any]) -> None:
    models = list_models()
    match = next((m for m in models if m["model_name"] == model_name), None)
    if match is None:
        available = sorted({m["model_name"] for m in models})
        raise ValueError(f"Unknown model {model_name!r}. Available: {available}")
    if match["task"] != task:
        raise ValueError(f"Model {model_name!r} is for task {match['task']!r}, not {task!r}.")
    schema = {p["name"]: p for p in match["parameters"]}
    for name, value in (params or {}).items():
        if name not in schema:
            raise ModelError(f"Unknown parameter {name!r} for {model_name!r}")
        s = schema[name]
        if value is None and not s.get("nullable"):
            raise ModelError(f"Parameter {name!r} does not accept None")
        if value is None:
            continue
        if s.get("choices") is not None and value not in s["choices"]:
            raise ModelError(f"Parameter {name!r} must be one of {s['choices']}, got {value!r}")
        if s.get("min_value") is not None and value < s["min_value"]:
            raise ModelError(f"Parameter {name!r} must be >= {s['min_value']}, got {value!r}")
        if s.get("max_value") is not None and value > s["max_value"]:
            raise ModelError(f"Parameter {name!r} must be <= {s['max_value']}, got {value!r}")


def _validate_grid_against_schema(model_name: str, task: str, grid: dict[str, list[Any]]) -> None:
    models = list_models()
    match = next((m for m in models if m["model_name"] == model_name), None)
    if match is None:
        return
    schema = {p["name"]: p for p in match["parameters"]}
    for name, values in grid.items():
        if name not in schema:
            raise ModelError(f"Unknown grid parameter {name!r} for {model_name!r}")
        s = schema[name]
        for value in values:
            if value is None and not s.get("nullable"):
                raise ModelError(f"Grid parameter {name!r} does not accept None")
            if value is None:
                continue
            if s.get("choices") is not None and value not in s["choices"]:
                raise ModelError(
                    f"Grid parameter {name!r} must be one of {s['choices']}, got {value!r}"
                )
            if s.get("min_value") is not None and value < s["min_value"]:
                raise ModelError(
                    f"Grid parameter {name!r} must be >= {s['min_value']}, got {value!r}"
                )
            if s.get("max_value") is not None and value > s["max_value"]:
                raise ModelError(
                    f"Grid parameter {name!r} must be <= {s['max_value']}, got {value!r}"
                )


def _alias_value(key: str, *values: Any) -> Any:
    present = [v for v in values if v is not None]
    if not present:
        return None
    first = present[0]
    for other in present[1:]:
        if other != first:
            raise ValueError(
                f"Conflicting values for {key!r}. Set it in only one place "
                f"(top-level, dataset/training/model block, or preprocessing)."
            )
    return first


def _parse_model_block(model: Any) -> dict[str, Any]:
    if model is None:
        return {}
    if isinstance(model, str):
        return {"model_name": model}
    if not isinstance(model, dict):
        raise ValueError("model must be a string or object.")
    out: dict[str, Any] = {}
    for key in ("model_name", "params", "task", "param_config"):
        if key in model:
            out[key] = model[key]
    unknown = set(model.keys()) - {"model_name", "params", "task", "param_config"}
    if unknown:
        raise ValueError(f"Unknown model keys: {sorted(unknown)}")
    return out


def config_to_train_request(data: dict[str, Any]) -> TrainRequest:
    unknown = set(data.keys()) - ALLOWED_TOP
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    raw_dataset = data.get("dataset")
    dataset_block: dict[str, Any] = {}
    dataset_name: str | None = None
    if isinstance(raw_dataset, str):
        dataset_name = raw_dataset
    elif isinstance(raw_dataset, dict):
        dataset_block = raw_dataset
        bad_dataset = set(dataset_block.keys()) - DATASET_KEYS
        if bad_dataset:
            raise ValueError(f"Unknown dataset keys: {sorted(bad_dataset)}")
        dataset_name = dataset_block.get("dataset")
    elif raw_dataset is not None:
        raise ValueError("dataset must be an object or a registered dataset name string.")

    training = data.get("training") or {}
    if data.get("training") is not None and not isinstance(training, dict):
        raise ValueError("training must be an object.")
    bad_training = set(training.keys()) - TRAINING_KEYS
    if bad_training:
        raise ValueError(f"Unknown training keys: {sorted(bad_training)}")

    pre = data.get("preprocessing")
    pre_ops: Any = None
    if pre is not None:
        if not isinstance(pre, dict):
            raise ValueError("preprocessing must be an object.")
        bad_pre = set(pre.keys()) - {"operations", "ops"}
        if bad_pre:
            raise ValueError(f"Unknown preprocessing keys: {sorted(bad_pre)}")
        if "operations" in pre and "ops" in pre and pre["operations"] != pre["ops"]:
            raise ValueError("Conflicting values for preprocessing operations/ops.")
        pre_ops = pre.get("operations", pre.get("ops"))

    model_block = _parse_model_block(data.get("model"))
    payload: dict[str, Any] = {}

    for key in ("source", "identifier", "split", "target"):
        value = _alias_value(
            key,
            dataset_block.get(key),
            training.get(key) if key in training else None,
            data.get(key),
        )
        if value is not None:
            payload[key] = value

    value = _alias_value(
        "dataset",
        dataset_name,
        training.get("dataset") if "dataset" in training else None,
    )
    if value is not None:
        payload["dataset"] = value

    for key in (
        "model_name",
        "task",
        "params",
        "param_config",
        "test_size",
        "validation_size",
        "stratify",
        "random_state",
        "log_mlflow",
        "experiment_name",
        "tracking_uri",
    ):
        value = _alias_value(
            key,
            training.get(key) if key in training else None,
            model_block.get(key) if key in model_block else None,
            data.get(key) if key in data else None,
        )
        if value is not None:
            payload[key] = value

    value = _alias_value(
        "operations",
        pre_ops,
        training.get("operations") if "operations" in training else None,
        data.get("operations") if "operations" in data else None,
    )
    if value is not None:
        payload["operations"] = value

    for key in ("model_name", "task", "params", "param_config"):
        if key in model_block:
            if key in payload and payload[key] != model_block[key]:
                raise ValueError(
                    f"Conflicting values for {key!r}. Set it in only one place "
                    f"(top-level, dataset/training/model block, or preprocessing)."
                )
            payload[key] = model_block[key]

    if "params" not in payload:
        for source in (training, model_block, data):
            if isinstance(source.get("params"), dict):
                payload["params"] = source["params"]
                break
    payload.setdefault("params", {})

    payload["operations"] = _validate_operations(payload.get("operations"))
    if "param_config" in payload:
        payload["param_config"] = _validate_param_config(payload["param_config"])

    params = payload.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("params must be an object.")
    payload["params"] = params

    model_name = payload.get("model_name", "logistic_regression")
    task = payload.get("task") or "classification"
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be classification or regression, got {task!r}")

    payload["model_name"] = model_name
    payload["task"] = task
    _validate_model(model_name, task, params)
    if payload.get("param_config"):
        _validate_grid_against_schema(model_name, task, payload["param_config"].get("grid") or {})

    if payload.get("source") and not payload.get("identifier"):
        raise ValueError("identifier is required when source is set.")
    if payload.get("identifier") and not payload.get("source"):
        raise ValueError("source is required when identifier is set.")
    if not payload.get("dataset") and not payload.get("identifier"):
        payload.setdefault("dataset", "titanic")

    for field, lo, hi in (
        ("test_size", 0.05, 0.5),
        ("validation_size", 0.0, 0.5),
    ):
        if field in payload and payload[field] is not None:
            try:
                val = float(payload[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be a number") from exc
            if val < lo or val > hi:
                raise ValueError(f"{field} must be between {lo} and {hi}, got {val}")
            payload[field] = val

    return TrainRequest(**payload)


def validate_config_text(text: str, fmt: str | None = None) -> dict[str, Any]:
    data = parse_config_text(text, fmt)
    req = config_to_train_request(data)
    return {
        "valid": True,
        "request": req.model_dump(),
        "normalized_yaml": yaml.safe_dump(req.model_dump(), sort_keys=False),
        "normalized_json": json.dumps(req.model_dump(), indent=2),
    }
