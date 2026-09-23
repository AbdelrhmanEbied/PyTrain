import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from models.sklearn import SklearnModel
from models.xgboost import XGBoostModel
from training.base import BaseTrainer
from training.estimator import EstimatorTrainer
from training.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS, compute_metrics
from training.param_config import ParamConfig
from training.results import EvalResult, TrainResult
from training.sklearn import SklearnTrainer
from training.xgboost import XGBoostTrainer


@pytest.fixture
def iris_data():
    data = load_iris()
    return data.data, data.target


@pytest.fixture
def diabetes_data():
    data = load_diabetes()
    return data.data, data.target


class TestMetrics:
    def test_classification_metrics(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        result = compute_metrics("classification", y_true, y_pred)
        assert "accuracy" in result
        assert "f1" in result
        assert "precision" in result
        assert "recall" in result
        assert all(isinstance(v, float) for v in result.values())
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_regression_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1])
        result = compute_metrics("regression", y_true, y_pred)
        assert "mse" in result
        assert "rmse" in result
        assert "mae" in result
        assert "r2" in result
        assert result["mse"] >= 0
        assert result["rmse"] >= 0

    def test_unknown_task(self):
        with pytest.raises(ValueError, match="Unknown task"):
            compute_metrics("clustering", np.array([1]), np.array([1]))

    def test_classification_metric_dicts(self):
        assert "accuracy" in CLASSIFICATION_METRICS
        assert "f1" in CLASSIFICATION_METRICS

    def test_regression_metric_dicts(self):
        assert "mse" in REGRESSION_METRICS
        assert "r2" in REGRESSION_METRICS

    def test_roc_auc_with_proba(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])
        result = compute_metrics("classification", y_true, y_pred, y_proba=y_proba)
        assert "roc_auc" in result


class TestResults:
    def test_eval_result_fields(self):
        er = EvalResult(
            metrics={"accuracy": 0.9},
            predictions=np.array([1, 0]),
            y_true=np.array([1, 0]),
            eval_time=0.1,
        )
        assert er.metrics["accuracy"] == 0.9
        assert er.eval_time == 0.1

    def test_train_result_fields(self):
        tr = TrainResult(
            model=None,
            metrics={"accuracy": 0.95},
            predictions=np.array([1, 0]),
            y_true=np.array([1, 0]),
            train_time=0.5,
            model_info={"model_name": "test"},
            params={"C": 1.0},
        )
        assert tr.eval_result is None
        assert tr.params["C"] == 1.0

    def test_train_result_with_eval(self):
        er = EvalResult(
            metrics={"accuracy": 0.8},
            predictions=np.array([1]),
            y_true=np.array([1]),
            eval_time=0.1,
        )
        tr = TrainResult(
            model=None,
            metrics={},
            predictions=np.array([1]),
            y_true=np.array([1]),
            train_time=0.1,
            model_info={},
            params={},
            eval_result=er,
        )
        assert tr.eval_result is not None
        assert tr.eval_result.metrics["accuracy"] == 0.8


class TestSklearnTrainerClassification:
    def test_train(self, iris_data):
        X, y = iris_data
        model = SklearnModel("logistic_regression", "classification")
        trainer = SklearnTrainer(model, "classification")
        result = trainer.train(X, y)

        assert isinstance(result, TrainResult)
        assert isinstance(result.model, LogisticRegression)
        assert "accuracy" in result.metrics
        assert result.train_time > 0
        assert len(result.predictions) == len(y)
        assert result.model_info["model_name"] == "logistic_regression"

    def test_train_with_validation(self, iris_data):
        X, y = iris_data
        model = SklearnModel("logistic_regression", "classification")
        trainer = SklearnTrainer(model, "classification")
        result = trainer.train(X, y, X_val=X, y_val=y)

        assert result.eval_result is not None
        assert "accuracy" in result.eval_result.metrics

    def test_evaluate(self, iris_data):
        X, y = iris_data
        model = SklearnModel("logistic_regression", "classification")
        trainer = SklearnTrainer(model, "classification")
        trainer.train(X, y)
        eval_result = trainer.evaluate(X, y)

        assert isinstance(eval_result, EvalResult)
        assert "accuracy" in eval_result.metrics
        assert eval_result.eval_time > 0

    def test_predict(self, iris_data):
        X, y = iris_data
        model = SklearnModel("logistic_regression", "classification")
        trainer = SklearnTrainer(model, "classification")
        trainer.train(X, y)
        preds = trainer.predict(X)

        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)

    def test_predict_before_train_raises(self, iris_data):
        X, y = iris_data
        model = SklearnModel("logistic_regression", "classification")
        trainer = SklearnTrainer(model, "classification")
        with pytest.raises(RuntimeError, match="not been fitted"):
            trainer.predict(X)

    def test_predict_proba(self, iris_data):
        X, y = iris_data
        model = SklearnModel("logistic_regression", "classification")
        trainer = SklearnTrainer(model, "classification")
        trainer.train(X, y)
        proba = trainer.predict_proba(X)

        assert proba is not None
        assert proba.shape[0] == len(y)

    def test_train_with_params(self, iris_data):
        X, y = iris_data
        model = SklearnModel("logistic_regression", "classification")
        model.set_params(C=0.1, max_iter=500)
        trainer = SklearnTrainer(model, "classification")
        result = trainer.train(X, y)

        assert result.model.C == 0.1
        assert result.model.max_iter == 500
        assert result.params["C"] == 0.1

    def test_random_forest_classifier(self, iris_data):
        X, y = iris_data
        model = SklearnModel("random_forest_classifier", "classification")
        trainer = SklearnTrainer(model, "classification")
        result = trainer.train(X, y)

        assert isinstance(result.model, RandomForestClassifier)
        assert "accuracy" in result.metrics


class TestSklearnTrainerRegression:
    def test_train(self, diabetes_data):
        X, y = diabetes_data
        model = SklearnModel("linear_regression", "regression")
        trainer = SklearnTrainer(model, "regression")
        result = trainer.train(X, y)

        assert isinstance(result.model, LinearRegression)
        assert "rmse" in result.metrics
        assert "r2" in result.metrics

    def test_train_with_validation(self, diabetes_data):
        X, y = diabetes_data
        model = SklearnModel("linear_regression", "regression")
        trainer = SklearnTrainer(model, "regression")
        result = trainer.train(X, y, X_val=X, y_val=y)

        assert result.eval_result is not None
        assert "rmse" in result.eval_result.metrics


class TestXGBoostTrainer:
    def test_classification(self, iris_data):
        X, y = iris_data
        model = XGBoostModel("xgb_classifier", "classification")
        trainer = XGBoostTrainer(model, "classification")
        result = trainer.train(X, y)

        assert isinstance(result.model, XGBClassifier)
        assert "accuracy" in result.metrics
        assert result.train_time > 0

    def test_regression(self, diabetes_data):
        X, y = diabetes_data
        model = XGBoostModel("xgb_regressor", "regression")
        trainer = XGBoostTrainer(model, "regression")
        result = trainer.train(X, y)

        assert isinstance(result.model, XGBRegressor)
        assert "rmse" in result.metrics

    def test_train_with_params(self, iris_data):
        X, y = iris_data
        model = XGBoostModel("xgb_classifier", "classification")
        model.set_params(n_estimators=50, max_depth=3)
        trainer = XGBoostTrainer(model, "classification")
        result = trainer.train(X, y)

        assert result.model.n_estimators == 50
        assert result.model.max_depth == 3

    def test_evaluate(self, iris_data):
        X, y = iris_data
        model = XGBoostModel("xgb_classifier", "classification")
        trainer = XGBoostTrainer(model, "classification")
        trainer.train(X, y)
        eval_result = trainer.evaluate(X, y)

        assert isinstance(eval_result, EvalResult)
        assert "accuracy" in eval_result.metrics


class TestParamConfig:
    def test_defaults(self):
        pc = ParamConfig(grid={"C": [0.1, 1.0]})
        assert pc.cv == 5
        assert pc.scoring is None
        assert pc.refit is True
        assert pc.n_jobs is None
        assert pc.verbose == 0

    def test_empty_grid_raises(self):
        with pytest.raises(ValueError, match="grid must not be empty"):
            ParamConfig(grid={})

    def test_cv_too_small_raises(self):
        with pytest.raises(ValueError, match="cv must be >= 2"):
            ParamConfig(grid={"C": [1.0]}, cv=1)

    def test_empty_values_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            ParamConfig(grid={"C": []})


class TestGridSearch:
    def test_train_with_grid_search(self, iris_data):
        X, y = iris_data
        model = SklearnModel("logistic_regression", "classification")
        trainer = SklearnTrainer(model, "classification")
        result = trainer.train(
            X,
            y,
            param_config=ParamConfig(grid={"C": [0.01, 0.1, 1.0], "max_iter": [100, 500]}, cv=3),
        )

        assert result.best_params is not None
        assert "C" in result.best_params
        assert "max_iter" in result.best_params
        assert result.best_params["C"] in (0.01, 0.1, 1.0)
        assert result.cv_results is not None
        assert result.cv_results["cv"] == 3
        assert result.cv_results["n_candidates"] == 6
        assert 0.0 <= result.cv_results["best_score"] <= 1.0
        assert result.model.C == result.best_params["C"]
        assert result.model.max_iter == result.best_params["max_iter"]
        assert result.params["C"] == result.best_params["C"]

    def test_grid_search_with_validation(self, iris_data):
        X, y = iris_data
        model = SklearnModel("logistic_regression", "classification")
        trainer = SklearnTrainer(model, "classification")
        result = trainer.train(
            X,
            y,
            X_val=X,
            y_val=y,
            param_config=ParamConfig(grid={"C": [0.1, 1.0]}, cv=3),
        )
        assert result.eval_result is not None
        assert "accuracy" in result.eval_result.metrics

    def test_grid_search_random_forest(self, iris_data):
        X, y = iris_data
        model = SklearnModel("random_forest_classifier", "classification")
        trainer = SklearnTrainer(model, "classification")
        result = trainer.train(
            X,
            y,
            param_config=ParamConfig(grid={"n_estimators": [10, 50], "max_depth": [3, 5]}, cv=3),
        )
        assert result.best_params["n_estimators"] in (10, 50)
        assert result.best_params["max_depth"] in (3, 5)
        assert result.cv_results["n_candidates"] == 4

    def test_grid_search_xgboost(self, iris_data):
        X, y = iris_data
        model = XGBoostModel("xgb_classifier", "classification")
        trainer = XGBoostTrainer(model, "classification")
        result = trainer.train(
            X,
            y,
            param_config=ParamConfig(grid={"n_estimators": [10, 30], "max_depth": [2, 4]}, cv=3),
        )
        assert result.best_params["n_estimators"] in (10, 30)
        assert result.best_params["max_depth"] in (2, 4)
        assert result.model.n_estimators == result.best_params["n_estimators"]

    def test_grid_search_custom_scoring(self, iris_data):
        X, y = iris_data
        model = SklearnModel("logistic_regression", "classification")
        trainer = SklearnTrainer(model, "classification")
        result = trainer.train(
            X,
            y,
            param_config=ParamConfig(grid={"C": [0.1, 1.0]}, cv=3, scoring="f1_weighted"),
        )
        assert result.cv_results["scoring"] == "f1_weighted"

    def test_grid_search_regression(self, diabetes_data):
        X, y = diabetes_data
        model = SklearnModel("linear_regression", "regression")
        trainer = SklearnTrainer(model, "regression")
        result = trainer.train(
            X,
            y,
            param_config=ParamConfig(grid={"fit_intercept": [True, False]}, cv=3),
        )
        assert result.best_params["fit_intercept"] in (True, False)
        assert result.cv_results["scoring"] == "r2"

    def test_grid_search_invalid_param(self, iris_data):
        X, y = iris_data
        model = SklearnModel("logistic_regression", "classification")
        trainer = SklearnTrainer(model, "classification")
        with pytest.raises(Exception, match="Unknown parameters"):
            trainer.train(
                X,
                y,
                param_config=ParamConfig(grid={"bad_param": [1, 2]}, cv=3),
            )

    def test_grid_search_invalid_value(self, iris_data):
        X, y = iris_data
        model = SklearnModel("logistic_regression", "classification")
        trainer = SklearnTrainer(model, "classification")
        with pytest.raises(Exception, match="must be one of"):
            trainer.train(
                X,
                y,
                param_config=ParamConfig(grid={"solver": ["bad_solver"]}, cv=3),
            )

    def test_no_param_config_still_works(self, iris_data):
        X, y = iris_data
        model = SklearnModel("logistic_regression", "classification")
        trainer = SklearnTrainer(model, "classification")
        result = trainer.train(X, y)
        assert result.best_params is None
        assert result.cv_results is None


class TestInheritance:
    def test_sklearn_trainer_is_estimator(self):
        assert issubclass(SklearnTrainer, EstimatorTrainer)

    def test_xgboost_trainer_is_estimator(self):
        assert issubclass(XGBoostTrainer, EstimatorTrainer)

    def test_estimator_is_base(self):
        assert issubclass(EstimatorTrainer, BaseTrainer)

    def test_base_is_abstract(self):
        with pytest.raises(TypeError):
            BaseTrainer()

    def test_repr(self):
        trainer = SklearnTrainer(
            SklearnModel("logistic_regression", "classification"),
            "classification",
        )
        assert "SklearnTrainer" in repr(trainer)
