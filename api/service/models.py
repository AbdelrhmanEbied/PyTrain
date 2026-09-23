from __future__ import annotations

from typing import Any

from models.base import ParameterSchema, Task
from models.sklearn import SCHEMAS as SKLEARN_SCHEMAS
from models.sklearn import SKLEARN_MODELS, SklearnModel
from models.xgboost import SCHEMAS as XGB_SCHEMAS
from models.xgboost import XGBOOST_MODELS, XGBoostModel
from training.param_config import ParamConfig
from training.sklearn import SklearnTrainer
from training.xgboost import XGBoostTrainer


def _schema_to_dict(schema: ParameterSchema) -> dict[str, Any]:
    return {
        "name": schema.name,
        "type": schema.type,
        "default": schema.default,
        "choices": schema.choices,
        "min_value": schema.min_value,
        "max_value": schema.max_value,
        "nullable": schema.nullable,
        "description": schema.description,
        "group": schema.group,
        "advanced": schema.advanced,
    }


def list_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for task, registry in SKLEARN_MODELS.items():
        for name in registry:
            schema = SKLEARN_SCHEMAS.get(name, {})
            models.append(
                {
                    "model_name": name,
                    "task": task,
                    "framework": "scikit-learn",
                    "parameters": [_schema_to_dict(s) for s in schema.values()],
                }
            )
    for task, registry in XGBOOST_MODELS.items():
        for name in registry:
            schema = XGB_SCHEMAS.get(name, {})
            models.append(
                {
                    "model_name": name,
                    "task": task,
                    "framework": "xgboost",
                    "parameters": [_schema_to_dict(s) for s in schema.values()],
                }
            )
    return models


def make_model(model_name: str, task: Task, params: dict[str, Any] | None = None):
    if model_name in SklearnModel.list_available_models():
        return SklearnModel(model_name, task, params)
    if model_name in XGBoostModel.list_available_models():
        return XGBoostModel(model_name, task, params)
    available = sorted(
        set(SklearnModel.list_available_models()) | set(XGBoostModel.list_available_models())
    )
    raise ValueError(f"Unknown model: {model_name!r}. Available: {available}")


def make_trainer(model: Any, task: Task):
    if model.framework == "scikit-learn":
        return SklearnTrainer(model, task)
    return XGBoostTrainer(model, task)


def make_param_config(raw: dict[str, Any]) -> ParamConfig:
    return ParamConfig(
        grid=raw["grid"],
        cv=raw.get("cv", 5),
        scoring=raw.get("scoring"),
        refit=raw.get("refit", True),
        n_jobs=raw.get("n_jobs"),
        verbose=raw.get("verbose", 0),
    )
