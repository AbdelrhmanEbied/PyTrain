import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from models.base import (
    BaseModel,
    DependencyError,
    InvalidParameterError,
    InvalidTaskError,
    ParameterSchema,
    UnsupportedModelError,
)
from models.estimator import EstimatorModel
from models.sklearn import SklearnModel
from models.xgboost import XGBoostModel


class TestSklearnModelConstruction:
    def test_logistic_regression(self):
        m = SklearnModel("logistic_regression", "classification")
        assert m.model_name == "logistic_regression"
        assert m.task == "classification"
        assert m.framework == "scikit-learn"

    def test_random_forest_classifier(self):
        m = SklearnModel("random_forest_classifier", "classification")
        assert m.model_name == "random_forest_classifier"

    def test_linear_regression(self):
        m = SklearnModel("linear_regression", "regression")
        assert m.task == "regression"

    def test_random_forest_regressor(self):
        m = SklearnModel("random_forest_regressor", "regression")
        assert m.task == "regression"

    def test_invalid_model_name(self):
        with pytest.raises(UnsupportedModelError, match="Unknown model"):
            SklearnModel("nonexistent_model", "classification")

    def test_invalid_task(self):
        with pytest.raises(InvalidTaskError, match="not supported"):
            SklearnModel("logistic_regression", "bad_task")

    def test_model_wrong_task(self):
        with pytest.raises(UnsupportedModelError, match="Unknown model"):
            SklearnModel("logistic_regression", "regression")

    def test_repr(self):
        m = SklearnModel("logistic_regression", "classification")
        r = repr(m)
        assert "SklearnModel" in r
        assert "logistic_regression" in r

    def test_list_available_models(self):
        models = SklearnModel.list_available_models()
        assert "logistic_regression" in models
        assert "random_forest_classifier" in models
        assert "linear_regression" in models
        assert "random_forest_regressor" in models
        assert models == sorted(models)


class TestSklearnParams:
    def test_get_params_default(self):
        m = SklearnModel("logistic_regression", "classification")
        params = m.get_params()
        assert params == {}

    def test_set_params(self):
        m = SklearnModel("logistic_regression", "classification")
        m.set_params(C=0.5, max_iter=200)
        params = m.get_params()
        assert params["C"] == 0.5
        assert params["max_iter"] == 200

    def test_set_params_unknown_param(self):
        m = SklearnModel("logistic_regression", "classification")
        with pytest.raises(InvalidParameterError, match="Unknown parameters"):
            m.set_params(bad_param=1)

    def test_set_params_none_not_allowed(self):
        m = SklearnModel("logistic_regression", "classification")
        with pytest.raises(InvalidParameterError, match="does not accept None"):
            m.set_params(C=None)

    def test_set_params_none_nullable(self):
        m = SklearnModel("logistic_regression", "classification")
        m.set_params(random_state=None)
        assert m.get_params()["random_state"] is None

    def test_set_params_choices(self):
        m = SklearnModel("logistic_regression", "classification")
        m.set_params(solver="liblinear")
        assert m.get_params()["solver"] == "liblinear"

    def test_set_params_invalid_choice(self):
        m = SklearnModel("logistic_regression", "classification")
        with pytest.raises(InvalidParameterError, match="must be one of"):
            m.set_params(solver="bad_solver")

    def test_set_params_min_value(self):
        m = SklearnModel("logistic_regression", "classification")
        with pytest.raises(InvalidParameterError, match="must be >="):
            m.set_params(C=-1.0)

    def test_set_params_returns_copy(self):
        m = SklearnModel("logistic_regression", "classification")
        m.set_params(C=1.0)
        params = m.get_params()
        params["C"] = 999
        assert m.get_params()["C"] == 1.0


class TestSklearnBuild:
    def test_build_logistic_regression(self):
        m = SklearnModel("logistic_regression", "classification")
        estimator = m.build()
        assert isinstance(estimator, LogisticRegression)

    def test_build_random_forest_classifier(self):
        m = SklearnModel("random_forest_classifier", "classification")
        estimator = m.build()
        assert isinstance(estimator, RandomForestClassifier)

    def test_build_linear_regression(self):
        m = SklearnModel("linear_regression", "regression")
        estimator = m.build()
        assert isinstance(estimator, LinearRegression)

    def test_build_random_forest_regressor(self):
        m = SklearnModel("random_forest_regressor", "regression")
        estimator = m.build()
        assert isinstance(estimator, RandomForestRegressor)

    def test_build_with_params(self):
        m = SklearnModel("logistic_regression", "classification")
        m.set_params(C=0.5, max_iter=200)
        estimator = m.build()
        assert isinstance(estimator, LogisticRegression)
        assert estimator.C == 0.5
        assert estimator.max_iter == 200


class TestSklearnSchema:
    def test_get_parameter_schema_logistic(self):
        m = SklearnModel("logistic_regression", "classification")
        schema = m.get_parameter_schema()
        assert isinstance(schema, list)
        assert len(schema) > 0
        names = [s.name for s in schema]
        assert "C" in names
        assert "max_iter" in names
        assert "solver" in names

    def test_schema_entry_fields(self):
        m = SklearnModel("logistic_regression", "classification")
        schema = m.get_parameter_schema()
        c_schema = next(s for s in schema if s.name == "C")
        assert isinstance(c_schema, ParameterSchema)
        assert c_schema.type == "float"
        assert c_schema.default == 1.0
        assert c_schema.min_value == 0.0

    def test_schema_choices(self):
        m = SklearnModel("logistic_regression", "classification")
        schema = m.get_parameter_schema()
        solver_schema = next(s for s in schema if s.name == "solver")
        assert solver_schema.choices is not None
        assert "lbfgs" in solver_schema.choices

    def test_schema_nullable(self):
        m = SklearnModel("logistic_regression", "classification")
        schema = m.get_parameter_schema()
        rs_schema = next(s for s in schema if s.name == "random_state")
        assert rs_schema.nullable is True


class TestXGBoostModelConstruction:
    def test_xgb_classifier(self):
        m = XGBoostModel("xgb_classifier", "classification")
        assert m.model_name == "xgb_classifier"
        assert m.framework == "xgboost"

    def test_xgb_regressor(self):
        m = XGBoostModel("xgb_regressor", "regression")
        assert m.model_name == "xgb_regressor"

    def test_invalid_model_name(self):
        with pytest.raises(UnsupportedModelError, match="Unknown model"):
            XGBoostModel("nonexistent", "classification")

    def test_invalid_task(self):
        with pytest.raises(InvalidTaskError, match="not supported"):
            XGBoostModel("xgb_classifier", "bad_task")

    def test_model_wrong_task(self):
        with pytest.raises(UnsupportedModelError, match="Unknown model"):
            XGBoostModel("xgb_classifier", "regression")

    def test_repr(self):
        m = XGBoostModel("xgb_classifier", "classification")
        r = repr(m)
        assert "XGBoostModel" in r
        assert "xgb_classifier" in r

    def test_list_available_models(self):
        models = XGBoostModel.list_available_models()
        assert "xgb_classifier" in models
        assert "xgb_regressor" in models
        assert models == sorted(models)


class TestXGBoostParams:
    def test_get_params_default(self):
        m = XGBoostModel("xgb_classifier", "classification")
        assert m.get_params() == {}

    def test_set_params(self):
        m = XGBoostModel("xgb_classifier", "classification")
        m.set_params(n_estimators=50, max_depth=3)
        params = m.get_params()
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 3

    def test_set_params_unknown(self):
        m = XGBoostModel("xgb_classifier", "classification")
        with pytest.raises(InvalidParameterError, match="Unknown parameters"):
            m.set_params(bad=1)

    def test_set_params_min_value(self):
        m = XGBoostModel("xgb_classifier", "classification")
        with pytest.raises(InvalidParameterError, match="must be >="):
            m.set_params(n_estimators=0)

    def test_set_params_max_value(self):
        m = XGBoostModel("xgb_classifier", "classification")
        with pytest.raises(InvalidParameterError, match="must be <="):
            m.set_params(learning_rate=2.0)

    def test_set_params_none_not_allowed(self):
        m = XGBoostModel("xgb_classifier", "classification")
        with pytest.raises(InvalidParameterError, match="does not accept None"):
            m.set_params(max_depth=None)

    def test_set_params_none_nullable(self):
        m = XGBoostModel("xgb_classifier", "classification")
        m.set_params(random_state=None)
        assert m.get_params()["random_state"] is None


class TestXGBoostBuild:
    def test_build_xgb_classifier(self):
        m = XGBoostModel("xgb_classifier", "classification")
        estimator = m.build()
        assert isinstance(estimator, XGBClassifier)

    def test_build_xgb_regressor(self):
        m = XGBoostModel("xgb_regressor", "regression")
        estimator = m.build()
        assert isinstance(estimator, XGBRegressor)

    def test_build_with_params(self):
        m = XGBoostModel("xgb_classifier", "classification")
        m.set_params(n_estimators=50, max_depth=3)
        estimator = m.build()
        assert isinstance(estimator, XGBClassifier)
        assert estimator.n_estimators == 50
        assert estimator.max_depth == 3


class TestXGBoostSchema:
    def test_get_parameter_schema(self):
        m = XGBoostModel("xgb_classifier", "classification")
        schema = m.get_parameter_schema()
        assert isinstance(schema, list)
        names = [s.name for s in schema]
        assert "n_estimators" in names
        assert "max_depth" in names
        assert "learning_rate" in names

    def test_schema_entry_fields(self):
        m = XGBoostModel("xgb_classifier", "classification")
        schema = m.get_parameter_schema()
        lr_schema = next(s for s in schema if s.name == "learning_rate")
        assert lr_schema.type == "float"
        assert lr_schema.min_value == 0.0
        assert lr_schema.max_value == 1.0


class TestModelInfo:
    def test_sklearn_model_info(self):
        m = SklearnModel("logistic_regression", "classification")
        m.set_params(C=0.5)
        info = m.get_model_info()
        assert info["model_name"] == "logistic_regression"
        assert info["task"] == "classification"
        assert info["framework"] == "scikit-learn"
        assert info["parameters"]["C"] == 0.5

    def test_xgboost_model_info(self):
        m = XGBoostModel("xgb_classifier", "classification")
        info = m.get_model_info()
        assert info["framework"] == "xgboost"
        assert info["task"] == "classification"


class TestInheritance:
    def test_sklearn_is_estimator(self):
        assert issubclass(SklearnModel, EstimatorModel)

    def test_xgboost_is_estimator(self):
        assert issubclass(XGBoostModel, EstimatorModel)

    def test_estimator_is_base(self):
        assert issubclass(EstimatorModel, BaseModel)

    def test_base_is_abstract(self):
        with pytest.raises(TypeError):
            BaseModel()

    def test_estimator_is_abstract(self):
        with pytest.raises(TypeError):
            EstimatorModel("test", "classification")


class TestDependencyError:
    def test_xgboost_missing_raises_dependency_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "xgboost":
                raise ImportError("No module named 'xgboost'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        from models.xgboost import _import_xgboost

        with pytest.raises(DependencyError, match="not installed"):
            _import_xgboost()


class TestParamSchema:
    def test_type_is_literal(self):
        s = ParameterSchema(name="x", type="int")
        assert s.type == "int"

    def test_default_none(self):
        s = ParameterSchema(name="x", type="int")
        assert s.default is None
        assert s.choices is None
        assert s.nullable is False
