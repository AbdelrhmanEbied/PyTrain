import pytest
from sklearn.datasets import load_iris

from mlflow_integration import MLflowTracker
from models.sklearn import SklearnModel
from training.sklearn import SklearnTrainer


@pytest.fixture
def iris_data():
    data = load_iris()
    return data.data, data.target


@pytest.fixture
def tracker(tmp_path):
    return MLflowTracker(
        experiment_name="test_experiment",
        tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )


@pytest.fixture
def train_result(iris_data):
    X, y = iris_data
    model = SklearnModel("logistic_regression", "classification")
    trainer = SklearnTrainer(model, "classification")
    return trainer.train(X, y, X_val=X, y_val=y)


class TestTrackerConstruction:
    def test_default_experiment(self, tmp_path):
        t = MLflowTracker(
            experiment_name="default_test",
            tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        )
        assert t.experiment_name == "default_test"

    def test_custom_tracking_uri(self, tmp_path):
        uri = f"sqlite:///{tmp_path}/custom.db"
        t = MLflowTracker(tracking_uri=uri)
        assert t.experiment_name == "pytrain"

    def test_ui_url(self, tracker):
        url = tracker.ui_url
        assert isinstance(url, str)
        assert len(url) > 0


class TestTrackerRuns:
    def test_start_and_end_run(self, tracker):
        tracker.start_run(run_name="test_run")
        tracker.end_run()

    def test_context_manager(self, tracker):
        with tracker as t:
            assert isinstance(t, MLflowTracker)

    def test_context_manager_logs_params(self, tracker):
        with tracker:
            tracker.log_params({"lr": 0.01})
            tracker.log_metrics({"accuracy": 0.95})


class TestTrackerLogging:
    def test_log_params(self, tracker):
        with tracker:
            tracker.log_params({"C": 1.0, "max_iter": 100})

    def test_log_metrics(self, tracker):
        with tracker:
            tracker.log_metrics({"accuracy": 0.95, "f1": 0.94})

    def test_log_model(self, tracker, train_result):
        with tracker:
            tracker.log_model(train_result.model)

    def test_log_train_result(self, tracker, train_result):
        with tracker:
            tracker.log_train_result(train_result)

    def test_log_train_result_with_eval(self, tracker, train_result):
        assert train_result.eval_result is not None
        with tracker:
            tracker.log_train_result(train_result)

    def test_log_train_result_fields(self, tracker, train_result):
        with tracker:
            tracker.log_params(train_result.params)
            tracker.log_metrics(train_result.metrics)
            tracker.log_metrics({"train_time": train_result.train_time})
            assert train_result.model_info["model_name"] == "logistic_regression"


class TestXGBoostLogging:
    def test_log_xgb_model(self, tracker, iris_data):
        from models.xgboost import XGBoostModel
        from training.xgboost import XGBoostTrainer

        X, y = iris_data
        model = XGBoostModel("xgb_classifier", "classification")
        model.set_params(n_estimators=10, max_depth=3, random_state=42)
        trainer = XGBoostTrainer(model, "classification")
        result = trainer.train(X, y)
        with tracker:
            tracker.log_train_result(result)
            assert tracker.active_run_id is not None

    def test_active_run_id_inside_context(self, tracker):
        with tracker:
            run_id = tracker.active_run_id
            assert run_id is not None
        assert tracker.active_run_id is None or run_id is not None
