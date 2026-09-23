from __future__ import annotations

from typing import Any

from models.base import DependencyError, ParameterSchema, Task
from models.estimator import EstimatorModel


def _import_xgboost():
    try:
        from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor

        return XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
    except ImportError as exc:
        raise DependencyError(
            "XGBoost is not installed. Install it with: pip install xgboost"
        ) from exc


XGBOOST_MODELS: dict[str, dict[str, str]] = {
    "classification": {
        "xgb_classifier": "XGBClassifier",
        "xgb_rf_classifier": "XGBRFClassifier",
    },
    "regression": {
        "xgb_regressor": "XGBRegressor",
        "xgb_rf_regressor": "XGBRFRegressor",
    },
}


def _base_params() -> dict[str, ParameterSchema]:
    return {
        "n_estimators": ParameterSchema(
            name="n_estimators",
            type="int",
            default=100,
            min_value=1,
            description="Number of boosting rounds (trees) to train",
            group="boosting",
        ),
        "max_depth": ParameterSchema(
            name="max_depth",
            type="int",
            default=6,
            min_value=0,
            description="Maximum depth of each tree; deeper trees fit more complex patterns",
            group="boosting",
        ),
        "learning_rate": ParameterSchema(
            name="learning_rate",
            type="float",
            default=0.3,
            min_value=0.0,
            max_value=1.0,
            description="Shrinkage applied to each boosting step; lower is more conservative",
            group="boosting",
        ),
        "subsample": ParameterSchema(
            name="subsample",
            type="float",
            default=1.0,
            min_value=0.0,
            max_value=1.0,
            description="Fraction of training rows sampled per tree",
            group="sampling",
            advanced=True,
        ),
        "colsample_bytree": ParameterSchema(
            name="colsample_bytree",
            type="float",
            default=1.0,
            min_value=0.0,
            max_value=1.0,
            description="Fraction of features sampled per tree",
            group="sampling",
            advanced=True,
        ),
        "reg_lambda": ParameterSchema(
            name="reg_lambda",
            type="float",
            default=1.0,
            min_value=0.0,
            description="L2 regularization term on weights",
            group="regularization",
            advanced=True,
        ),
        "reg_alpha": ParameterSchema(
            name="reg_alpha",
            type="float",
            default=0.0,
            min_value=0.0,
            description="L1 regularization term on weights",
            group="regularization",
            advanced=True,
        ),
        "random_state": ParameterSchema(
            name="random_state",
            type="int",
            default=None,
            nullable=True,
            description="Seed for reproducible results; None uses random seed",
            group="reproducibility",
            advanced=True,
        ),
    }


_XGB_PARAMS = _base_params()

_XGB_RF_PARAMS = {
    "n_estimators": ParameterSchema(
        name="n_estimators",
        type="int",
        default=100,
        min_value=1,
        description="Number of trees in the random-forest ensemble",
        group="trees",
    ),
    "max_depth": ParameterSchema(
        name="max_depth",
        type="int",
        default=6,
        min_value=0,
        description="Maximum depth of each tree",
        group="trees",
    ),
    "learning_rate": ParameterSchema(
        name="learning_rate",
        type="float",
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        description="Learning rate (typically 1.0 for random-forest mode)",
        group="boosting",
        advanced=True,
    ),
    "subsample": ParameterSchema(
        name="subsample",
        type="float",
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        description="Fraction of training rows sampled per tree",
        group="sampling",
        advanced=True,
    ),
    "colsample_bynode": ParameterSchema(
        name="colsample_bynode",
        type="float",
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        description="Fraction of features sampled per tree node",
        group="sampling",
        advanced=True,
    ),
    "reg_lambda": ParameterSchema(
        name="reg_lambda",
        type="float",
        default=1.0,
        min_value=0.0,
        description="L2 regularization term on weights",
        group="regularization",
        advanced=True,
    ),
    "random_state": ParameterSchema(
        name="random_state",
        type="int",
        default=None,
        nullable=True,
        description="Seed for reproducible results; None uses random seed",
        group="reproducibility",
        advanced=True,
    ),
}

SCHEMAS: dict[str, dict[str, ParameterSchema]] = {
    "xgb_classifier": _XGB_PARAMS,
    "xgb_regressor": _XGB_PARAMS,
    "xgb_rf_classifier": _XGB_RF_PARAMS,
    "xgb_rf_regressor": _XGB_RF_PARAMS,
}


class XGBoostModel(EstimatorModel):
    framework = "xgboost"

    def __init__(self, model_name: str, task: Task, params: dict[str, Any] | None = None) -> None:
        super().__init__(model_name, task, params)

    @classmethod
    def list_available_models(cls) -> list[str]:
        return sorted({name for models in XGBOOST_MODELS.values() for name in models})

    def _model_registry(self) -> dict[str, Any]:
        task_models = XGBOOST_MODELS.get(self.task, {})
        XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor = _import_xgboost()
        class_map = {
            "XGBClassifier": XGBClassifier,
            "XGBRegressor": XGBRegressor,
            "XGBRFClassifier": XGBRFClassifier,
            "XGBRFRegressor": XGBRFRegressor,
        }
        return {name: class_map[cls_name] for name, cls_name in task_models.items()}

    def _schema(self) -> dict[str, ParameterSchema]:
        return SCHEMAS.get(self.model_name, {})
