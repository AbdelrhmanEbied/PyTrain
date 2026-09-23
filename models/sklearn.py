from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from models.base import ParameterSchema, Task
from models.estimator import EstimatorModel

SKLEARN_MODELS: dict[str, dict[str, Any]] = {
    "classification": {
        "logistic_regression": LogisticRegression,
        "random_forest_classifier": RandomForestClassifier,
        "extra_trees_classifier": ExtraTreesClassifier,
        "gradient_boosting_classifier": GradientBoostingClassifier,
        "hist_gradient_boosting_classifier": HistGradientBoostingClassifier,
        "adaboost_classifier": AdaBoostClassifier,
        "bagging_classifier": BaggingClassifier,
        "decision_tree_classifier": DecisionTreeClassifier,
        "knn_classifier": KNeighborsClassifier,
        "svm_classifier": SVC,
        "gaussian_nb_classifier": GaussianNB,
        "sgd_classifier": SGDClassifier,
        "mlp_classifier": MLPClassifier,
    },
    "regression": {
        "linear_regression": LinearRegression,
        "random_forest_regressor": RandomForestRegressor,
        "extra_trees_regressor": ExtraTreesRegressor,
        "gradient_boosting_regressor": GradientBoostingRegressor,
        "hist_gradient_boosting_regressor": HistGradientBoostingRegressor,
        "adaboost_regressor": AdaBoostRegressor,
        "bagging_regressor": BaggingRegressor,
        "decision_tree_regressor": DecisionTreeRegressor,
        "knn_regressor": KNeighborsRegressor,
        "svm_regressor": SVR,
        "sgd_regressor": SGDRegressor,
        "mlp_regressor": MLPRegressor,
        "ridge_regression": Ridge,
        "lasso_regression": Lasso,
        "elasticnet_regression": ElasticNet,
    },
}


def _random_state() -> ParameterSchema:
    return ParameterSchema(
        name="random_state",
        type="int",
        default=None,
        nullable=True,
        description="Seed for reproducible results; None uses random seed",
        group="reproducibility",
        advanced=True,
    )


def _max_depth(default: Any = None, min_value: int | None = None) -> ParameterSchema:
    return ParameterSchema(
        name="max_depth",
        type="int",
        default=default,
        min_value=min_value,
        nullable=True,
        description="Maximum tree depth; None grows until leaves are pure",
        group="trees",
    )


def _n_estimators(default: int = 100) -> ParameterSchema:
    return ParameterSchema(
        name="n_estimators",
        type="int",
        default=default,
        min_value=1,
        description="Number of base estimators",
        group="trees",
    )


def _min_samples_split() -> ParameterSchema:
    return ParameterSchema(
        name="min_samples_split",
        type="float",
        default=2.0,
        min_value=2.0,
        description="Minimum samples required to split an internal node",
        group="trees",
        advanced=True,
    )


def _max_features() -> ParameterSchema:
    return ParameterSchema(
        name="max_features",
        type="str",
        default="sqrt",
        choices=["sqrt", "log2"],
        nullable=True,
        description="Number of features to consider for best split; None uses all",
        group="trees",
        advanced=True,
    )


_RANDOM_FOREST_PARAMS = {
    "n_estimators": _n_estimators(100),
    "max_depth": _max_depth(),
    "min_samples_split": _min_samples_split(),
    "max_features": _max_features(),
    "bootstrap": ParameterSchema(
        name="bootstrap",
        type="bool",
        default=True,
        description="Whether bootstrap samples are used when building trees",
        group="sampling",
        advanced=True,
    ),
    "random_state": _random_state(),
}

_EXTRA_TREES_PARAMS = dict(_RANDOM_FOREST_PARAMS)

_TREE_CLASSIFIER_PARAMS = {
    "criterion": ParameterSchema(
        name="criterion",
        type="str",
        default="gini",
        choices=["gini", "entropy", "log_loss"],
        description="Function to measure split quality",
        group="trees",
    ),
    "max_depth": _max_depth(),
    "min_samples_split": _min_samples_split(),
    "min_samples_leaf": ParameterSchema(
        name="min_samples_leaf",
        type="int",
        default=1,
        min_value=1,
        description="Minimum samples required at a leaf node",
        group="trees",
        advanced=True,
    ),
    "max_features": _max_features(),
    "random_state": _random_state(),
}

_TREE_REGRESSOR_PARAMS = {
    **{k: v for k, v in _TREE_CLASSIFIER_PARAMS.items() if k != "criterion"},
    "criterion": ParameterSchema(
        name="criterion",
        type="str",
        default="squared_error",
        choices=["squared_error", "friedman_mse", "absolute_error", "poisson"],
        description="Function to measure split quality",
        group="trees",
    ),
}

_GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": _n_estimators(100),
    "learning_rate": ParameterSchema(
        name="learning_rate",
        type="float",
        default=0.1,
        min_value=0.0,
        max_value=1.0,
        description="Shrinkage applied to each boosting step",
        group="boosting",
    ),
    "max_depth": _max_depth(default=3, min_value=1),
    "subsample": ParameterSchema(
        name="subsample",
        type="float",
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        description="Fraction of samples used to fit each base learner",
        group="sampling",
        advanced=True,
    ),
    "random_state": _random_state(),
}

_HIST_GB_PARAMS = {
    "max_iter": ParameterSchema(
        name="max_iter",
        type="int",
        default=100,
        min_value=1,
        description="Maximum number of boosting iterations",
        group="boosting",
    ),
    "learning_rate": ParameterSchema(
        name="learning_rate",
        type="float",
        default=0.1,
        min_value=0.0,
        max_value=1.0,
        description="Shrinkage applied to each boosting step",
        group="boosting",
    ),
    "max_leaf_nodes": ParameterSchema(
        name="max_leaf_nodes",
        type="int",
        default=31,
        min_value=2,
        description="Maximum number of leaves per tree",
        group="trees",
        advanced=True,
    ),
    "max_depth": _max_depth(),
    "min_samples_leaf": ParameterSchema(
        name="min_samples_leaf",
        type="int",
        default=20,
        min_value=1,
        description="Minimum samples required at a leaf node",
        group="trees",
        advanced=True,
    ),
    "l2_regularization": ParameterSchema(
        name="l2_regularization",
        type="float",
        default=0.0,
        min_value=0.0,
        description="L2 regularization term on leaf weights",
        group="regularization",
        advanced=True,
    ),
    "random_state": _random_state(),
}

_ADABOOST_PARAMS = {
    "n_estimators": _n_estimators(50),
    "learning_rate": ParameterSchema(
        name="learning_rate",
        type="float",
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        description="Weight applied to each classifier at each boosting iteration",
        group="boosting",
    ),
    "random_state": _random_state(),
}

_BAGGING_PARAMS = {
    "n_estimators": _n_estimators(10),
    "max_samples": ParameterSchema(
        name="max_samples",
        type="float",
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        description="Fraction of samples to draw for each base estimator",
        group="sampling",
        advanced=True,
    ),
    "max_features": ParameterSchema(
        name="max_features",
        type="float",
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        description="Fraction of features to draw for each base estimator",
        group="sampling",
        advanced=True,
    ),
    "bootstrap": ParameterSchema(
        name="bootstrap",
        type="bool",
        default=True,
        description="Whether samples are drawn with replacement",
        group="sampling",
        advanced=True,
    ),
    "n_jobs": ParameterSchema(
        name="n_jobs",
        type="int",
        default=None,
        nullable=True,
        description="Number of parallel jobs; None or 1 uses one core",
        group="optimization",
        advanced=True,
    ),
    "random_state": _random_state(),
}

_KNN_PARAMS = {
    "n_neighbors": ParameterSchema(
        name="n_neighbors",
        type="int",
        default=5,
        min_value=1,
        description="Number of neighbors to use",
        group="general",
    ),
    "weights": ParameterSchema(
        name="weights",
        type="str",
        default="uniform",
        choices=["uniform", "distance"],
        description="Weight function used in prediction",
        group="general",
    ),
    "p": ParameterSchema(
        name="p",
        type="int",
        default=2,
        choices=[1, 2],
        description="Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)",
        group="optimization",
        advanced=True,
    ),
    "metric": ParameterSchema(
        name="metric",
        type="str",
        default="minkowski",
        choices=["minkowski", "euclidean", "manhattan"],
        description="Distance metric",
        group="optimization",
        advanced=True,
    ),
}

_SVM_CLASSIFIER_PARAMS = {
    "C": ParameterSchema(
        name="C",
        type="float",
        default=1.0,
        min_value=0.0,
        description="Inverse regularization strength; smaller values regularize more",
        group="regularization",
    ),
    "kernel": ParameterSchema(
        name="kernel",
        type="str",
        default="rbf",
        choices=["linear", "rbf", "poly", "sigmoid"],
        description="Kernel type",
        group="general",
    ),
    "gamma": ParameterSchema(
        name="gamma",
        type="str",
        default="scale",
        choices=["scale", "auto"],
        description="Kernel coefficient",
        group="general",
        advanced=True,
    ),
    "probability": ParameterSchema(
        name="probability",
        type="bool",
        default=False,
        description="Whether to enable probability estimates",
        group="general",
        advanced=True,
    ),
    "random_state": _random_state(),
}

_SVM_REGRESSOR_PARAMS = {
    **{k: v for k, v in _SVM_CLASSIFIER_PARAMS.items() if k not in ("probability", "random_state")},
    "epsilon": ParameterSchema(
        name="epsilon",
        type="float",
        default=0.0,
        min_value=0.0,
        description="Epsilon in the epsilon-SVR model",
        group="regularization",
        advanced=True,
    ),
}

_SGD_CLASSIFIER_PARAMS = {
    "loss": ParameterSchema(
        name="loss",
        type="str",
        default="log_loss",
        choices=["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"],
        description="Loss function",
        group="optimization",
    ),
    "penalty": ParameterSchema(
        name="penalty",
        type="str",
        default="l2",
        choices=["l2", "l1", "elasticnet"],
        description="Regularization term",
        group="regularization",
    ),
    "alpha": ParameterSchema(
        name="alpha",
        type="float",
        default=0.0001,
        min_value=0.0,
        description="Constant that multiplies the regularization term",
        group="regularization",
    ),
    "max_iter": ParameterSchema(
        name="max_iter",
        type="int",
        default=1000,
        min_value=1,
        description="Maximum number of passes over the training data",
        group="optimization",
    ),
    "learning_rate": ParameterSchema(
        name="learning_rate",
        type="str",
        default="optimal",
        choices=["optimal", "constant", "invscaling", "adaptive"],
        description="Learning rate schedule",
        group="optimization",
        advanced=True,
    ),
    "random_state": _random_state(),
}

_SGD_REGRESSOR_PARAMS = {
    **dict(_SGD_CLASSIFIER_PARAMS),
    "loss": ParameterSchema(
        name="loss",
        type="str",
        default="squared_error",
        choices=["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
        description="Loss function",
        group="optimization",
    ),
}

_MLP_PARAMS = {
    "hidden_layer_sizes": ParameterSchema(
        name="hidden_layer_sizes",
        type="str",
        default="64,32",
        description="Comma-separated hidden layer sizes, e.g. 64,32",
        group="general",
    ),
    "activation": ParameterSchema(
        name="activation",
        type="str",
        default="relu",
        choices=["identity", "logistic", "tanh", "relu"],
        description="Activation function for the hidden layer",
        group="general",
    ),
    "solver": ParameterSchema(
        name="solver",
        type="str",
        default="adam",
        choices=["adam", "sgd", "lbfgs"],
        description="Solver for weight optimization",
        group="optimization",
    ),
    "alpha": ParameterSchema(
        name="alpha",
        type="float",
        default=0.0001,
        min_value=0.0,
        description="L2 regularization term",
        group="regularization",
    ),
    "learning_rate_init": ParameterSchema(
        name="learning_rate_init",
        type="float",
        default=0.001,
        min_value=0.0,
        description="Initial learning rate",
        group="optimization",
        advanced=True,
    ),
    "max_iter": ParameterSchema(
        name="max_iter",
        type="int",
        default=200,
        min_value=1,
        description="Maximum number of iterations",
        group="optimization",
    ),
    "random_state": _random_state(),
}

_LINEAR_REG_PARAMS = {
    "fit_intercept": ParameterSchema(
        name="fit_intercept",
        type="bool",
        default=True,
        description="Whether to calculate the intercept term",
        group="general",
    ),
}

_RIDGE_PARAMS = {
    **dict(_LINEAR_REG_PARAMS),
    "alpha": ParameterSchema(
        name="alpha",
        type="float",
        default=1.0,
        min_value=0.0,
        description="Regularization strength; higher is more regularized",
        group="regularization",
    ),
}

_LASSO_PARAMS = {
    **dict(_RIDGE_PARAMS),
    "max_iter": ParameterSchema(
        name="max_iter",
        type="int",
        default=1000,
        min_value=1,
        description="Maximum number of iterations",
        group="optimization",
        advanced=True,
    ),
    "random_state": _random_state(),
}

_ELASTICNET_PARAMS = {
    **dict(_LASSO_PARAMS),
    "l1_ratio": ParameterSchema(
        name="l1_ratio",
        type="float",
        default=0.5,
        min_value=0.0,
        max_value=1.0,
        description="Mix of L1/L2 regularization; 0=L2, 1=L1",
        group="regularization",
    ),
}

_GNB_PARAMS = {
    "var_smoothing": ParameterSchema(
        name="var_smoothing",
        type="float",
        default=1e-9,
        min_value=0.0,
        description="Portion of the largest variance of all features added to variances",
        group="regularization",
        advanced=True,
    ),
}


def _parse_hidden_layer_sizes(value: Any) -> Any:
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace("(", "").replace(")", "").split(",") if p.strip()]
        return tuple(int(p) for p in parts)
    if isinstance(value, list):
        return tuple(value)
    return value


SCHEMAS: dict[str, dict[str, ParameterSchema]] = {
    "logistic_regression": {
        "C": ParameterSchema(
            name="C",
            type="float",
            default=1.0,
            min_value=0.0,
            description="Inverse regularization strength; smaller values regularize more",
            group="regularization",
        ),
        "max_iter": ParameterSchema(
            name="max_iter",
            type="int",
            default=100,
            min_value=1,
            description="Maximum iterations for the solver to converge",
            group="optimization",
        ),
        "solver": ParameterSchema(
            name="solver",
            type="str",
            default="lbfgs",
            choices=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
            description="Optimization algorithm used to fit the coefficients",
            group="optimization",
            advanced=True,
        ),
        "penalty": ParameterSchema(
            name="penalty",
            type="str",
            default="l2",
            choices=["l1", "l2", "elasticnet", "none"],
            description="Regularization term",
            group="regularization",
            advanced=True,
        ),
        "random_state": _random_state(),
    },
    "random_forest_classifier": _RANDOM_FOREST_PARAMS,
    "random_forest_regressor": _RANDOM_FOREST_PARAMS,
    "extra_trees_classifier": _EXTRA_TREES_PARAMS,
    "extra_trees_regressor": _EXTRA_TREES_PARAMS,
    "gradient_boosting_classifier": _GRADIENT_BOOSTING_PARAMS,
    "gradient_boosting_regressor": _GRADIENT_BOOSTING_PARAMS,
    "hist_gradient_boosting_classifier": _HIST_GB_PARAMS,
    "hist_gradient_boosting_regressor": _HIST_GB_PARAMS,
    "adaboost_classifier": _ADABOOST_PARAMS,
    "adaboost_regressor": _ADABOOST_PARAMS,
    "bagging_classifier": _BAGGING_PARAMS,
    "bagging_regressor": _BAGGING_PARAMS,
    "decision_tree_classifier": _TREE_CLASSIFIER_PARAMS,
    "decision_tree_regressor": _TREE_REGRESSOR_PARAMS,
    "knn_classifier": _KNN_PARAMS,
    "knn_regressor": _KNN_PARAMS,
    "svm_classifier": _SVM_CLASSIFIER_PARAMS,
    "svm_regressor": _SVM_REGRESSOR_PARAMS,
    "gaussian_nb_classifier": _GNB_PARAMS,
    "sgd_classifier": _SGD_CLASSIFIER_PARAMS,
    "sgd_regressor": _SGD_REGRESSOR_PARAMS,
    "mlp_classifier": _MLP_PARAMS,
    "mlp_regressor": _MLP_PARAMS,
    "linear_regression": _LINEAR_REG_PARAMS,
    "ridge_regression": _RIDGE_PARAMS,
    "lasso_regression": _LASSO_PARAMS,
    "elasticnet_regression": _ELASTICNET_PARAMS,
}


class SklearnModel(EstimatorModel):
    framework = "scikit-learn"

    def __init__(self, model_name: str, task: Task, params: dict[str, Any] | None = None) -> None:
        super().__init__(model_name, task, params)

    @classmethod
    def list_available_models(cls) -> list[str]:
        return sorted({name for models in SKLEARN_MODELS.values() for name in models})

    def _model_registry(self) -> dict[str, Any]:
        return SKLEARN_MODELS.get(self.task, {})

    def _schema(self) -> dict[str, ParameterSchema]:
        return SCHEMAS.get(self.model_name, {})

    def build(self) -> Any:
        params = dict(self._params)
        if self.model_name in {"mlp_classifier", "mlp_regressor"}:
            if "hidden_layer_sizes" in params:
                params["hidden_layer_sizes"] = _parse_hidden_layer_sizes(
                    params["hidden_layer_sizes"]
                )
        return self._model_registry()[self.model_name](**params)
