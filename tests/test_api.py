import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}


class TestIndex:
    def test_index(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "PyTrain" in res.text


class TestModels:
    def test_list_models(self, client):
        res = client.get("/api/models")
        assert res.status_code == 200
        data = res.json()
        names = [m["model_name"] for m in data]
        assert "logistic_regression" in names
        assert "xgb_classifier" in names
        assert all("parameters" in m for m in data)
        assert all("task" in m for m in data)
        assert all("framework" in m for m in data)

    def test_schema_fields(self, client):
        data = client.get("/api/models").json()
        lr = next(m for m in data if m["model_name"] == "logistic_regression")
        param_names = [p["name"] for p in lr["parameters"]]
        assert "C" in param_names
        c = next(p for p in lr["parameters"] if p["name"] == "C")
        assert c["type"] == "float"
        assert c["min_value"] == 0.0


class TestDatasets:
    def test_list_datasets(self, client):
        res = client.get("/api/datasets")
        assert res.status_code == 200
        names = [d["name"] for d in res.json()]
        assert "titanic" in names
        assert "iris" in names

    def test_dataset_preview(self, client):
        res = client.get("/api/datasets/iris")
        assert res.status_code == 200
        data = res.json()
        assert data["rows"] == 150
        assert "target" in data["columns"]
        assert len(data["preview"]) > 0

    def test_unknown_dataset(self, client):
        res = client.get("/api/datasets/nope")
        assert res.status_code == 400


class TestTrainIris:
    def test_train_logistic(self, client):
        res = client.post(
            "/api/train",
            json={
                "dataset": "iris",
                "target": "target",
                "model_name": "logistic_regression",
                "task": "classification",
                "params": {"max_iter": 500},
                "test_size": 0.2,
                "validation_size": 0.1,
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["test_metrics"]["accuracy"] > 0.85
        assert data["train_time"] > 0
        assert data["n_features"] > 0
        assert data["model_info"]["model_name"] == "logistic_regression"
        assert data["logged_to_mlflow"] is False

    def test_train_with_grid(self, client):
        res = client.post(
            "/api/train",
            json={
                "dataset": "iris",
                "target": "target",
                "model_name": "logistic_regression",
                "task": "classification",
                "params": {"max_iter": 500},
                "param_config": {"grid": {"C": [0.1, 1.0]}, "cv": 3},
                "test_size": 0.2,
                "validation_size": 0.1,
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["best_params"] is not None
        assert data["best_params"]["C"] in (0.1, 1.0)
        assert data["cv_results"]["cv"] == 3
        assert data["cv_results"]["n_candidates"] == 2

    def test_train_random_forest(self, client):
        res = client.post(
            "/api/train",
            json={
                "dataset": "iris",
                "target": "target",
                "model_name": "random_forest_classifier",
                "task": "classification",
                "params": {"n_estimators": 50, "random_state": 42},
                "test_size": 0.2,
                "validation_size": 0.1,
            },
        )
        assert res.status_code == 200, res.text
        assert res.json()["test_metrics"]["accuracy"] > 0.85


class TestTrainTitanic:
    def test_train_titanic(self, client):
        res = client.post(
            "/api/train",
            json={
                "dataset": "titanic",
                "target": "survived",
                "model_name": "logistic_regression",
                "task": "classification",
                "params": {"max_iter": 500},
                "test_size": 0.2,
                "validation_size": 0.1,
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["test_metrics"]["accuracy"] > 0.7
        assert data["n_train"] + data["n_val"] + data["n_test"] > 1000

    def test_train_titanic_xgboost(self, client):
        res = client.post(
            "/api/train",
            json={
                "dataset": "titanic",
                "target": "survived",
                "model_name": "xgb_classifier",
                "task": "classification",
                "params": {"n_estimators": 50, "max_depth": 3, "random_state": 42},
                "test_size": 0.2,
                "validation_size": 0.1,
            },
        )
        assert res.status_code == 200, res.text
        assert res.json()["test_metrics"]["accuracy"] > 0.7


class TestTrainRegression:
    def test_diabetes(self, client):
        res = client.post(
            "/api/train",
            json={
                "dataset": "diabetes",
                "target": "target",
                "model_name": "linear_regression",
                "task": "regression",
                "test_size": 0.2,
                "validation_size": 0.1,
            },
        )
        assert res.status_code == 200, res.text
        assert "r2" in res.json()["test_metrics"]
        assert "rmse" in res.json()["test_metrics"]


class TestErrors:
    def test_unknown_model(self, client):
        res = client.post(
            "/api/train",
            json={
                "dataset": "iris",
                "model_name": "nope",
                "task": "classification",
            },
        )
        assert res.status_code == 400

    def test_unknown_dataset(self, client):
        res = client.post(
            "/api/train",
            json={
                "dataset": "nope",
                "model_name": "logistic_regression",
                "task": "classification",
            },
        )
        assert res.status_code == 400

    def test_unknown_target(self, client):
        res = client.post(
            "/api/train",
            json={
                "dataset": "iris",
                "target": "nope",
                "model_name": "logistic_regression",
                "task": "classification",
            },
        )
        assert res.status_code == 400

    def test_invalid_grid_param(self, client):
        res = client.post(
            "/api/train",
            json={
                "dataset": "iris",
                "model_name": "logistic_regression",
                "task": "classification",
                "param_config": {"grid": {"bad_param": [1]}, "cv": 3},
            },
        )
        assert res.status_code == 400

    def test_wrong_task_model(self, client):
        res = client.post(
            "/api/train",
            json={
                "dataset": "iris",
                "model_name": "logistic_regression",
                "task": "regression",
            },
        )
        assert res.status_code == 400


class TestDatasetSources:
    def test_list_includes_source_fields(self, client):
        data = client.get("/api/datasets").json()
        iris = next(d for d in data if d["name"] == "iris")
        assert iris["source"] == "sklearn"
        assert iris["identifier"] == "iris"

    def test_preview_by_ref(self, client):
        res = client.post(
            "/api/datasets/preview",
            json={"source": "sklearn", "identifier": "iris", "n_rows": 5},
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["rows"] == 150
        assert len(data["preview"]) == 5

    def test_preview_local_path(self, client, csv_file):
        res = client.post(
            "/api/datasets/preview",
            json={"source": "local", "identifier": str(csv_file)},
        )
        assert res.status_code == 200, res.text
        assert res.json()["rows"] == 3

    def test_preview_local_missing(self, client, tmp_path):
        res = client.post(
            "/api/datasets/preview",
            json={"source": "local", "identifier": str(tmp_path / "nope.csv")},
        )
        assert res.status_code == 404

    def test_train_with_source(self, client):
        res = client.post(
            "/api/train",
            json={
                "source": "sklearn",
                "identifier": "iris",
                "target": "target",
                "model_name": "logistic_regression",
                "task": "classification",
                "params": {"max_iter": 500},
            },
        )
        assert res.status_code == 200, res.text
        assert res.json()["test_metrics"]["accuracy"] > 0.85

    def test_upload_dataset(self, client, csv_file):
        content = csv_file.read_bytes()
        res = client.post(
            "/api/datasets/upload",
            files={"file": ("sample.csv", content, "text/csv")},
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["name"] == "sample"
        assert data["rows"] == 3

        res = client.get("/api/datasets/sample")
        assert res.status_code == 200
        assert res.json()["rows"] == 3

    def test_upload_unsupported_format(self, client):
        res = client.post(
            "/api/datasets/upload",
            files={"file": ("data.xyz", b"hello", "application/octet-stream")},
        )
        assert res.status_code == 400


class TestPreprocess:
    def test_no_operations(self, client):
        res = client.post(
            "/api/preprocess",
            json={"source": "sklearn", "identifier": "iris", "operations": []},
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["rows"] == 150
        assert data["operations"] == []
        assert data["before_shape"] == data["after_shape"]

    def test_apply_operations(self, client):
        res = client.post(
            "/api/preprocess",
            json={
                "source": "sklearn",
                "identifier": "iris",
                "operations": [
                    {"operation": "rename_columns", "mapping": {"target": "label"}},
                    {"operation": "shuffle", "random_state": 1},
                ],
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert "label" in data["columns"]
        assert "target" not in data["columns"]
        assert len(data["operations"]) == 2

    def test_unknown_operation(self, client):
        res = client.post(
            "/api/preprocess",
            json={
                "source": "sklearn",
                "identifier": "iris",
                "operations": [{"operation": "explode"}],
            },
        )
        assert res.status_code == 400


class TestJobs:
    def test_train_job_completes(self, client):
        res = client.post(
            "/api/jobs/train",
            json={
                "source": "sklearn",
                "identifier": "iris",
                "target": "target",
                "model_name": "logistic_regression",
                "task": "classification",
                "params": {"max_iter": 500},
            },
        )
        assert res.status_code == 200, res.text
        job_id = res.json()["job_id"]

        data = None
        for _ in range(120):
            status = client.get(f"/api/jobs/{job_id}").json()
            if status["status"] in ("done", "error"):
                data = status
                break
            import time

            time.sleep(0.25)

        assert data is not None
        assert data["status"] == "done", data.get("error")
        assert data["result"]["test_metrics"]["accuracy"] > 0.85

    def test_job_not_found(self, client):
        res = client.get("/api/jobs/nope")
        assert res.status_code == 404


class TestRuns:
    def test_log_and_list_runs(self, client, tmp_path):
        uri = f"sqlite:///{tmp_path}/runs.db"
        res = client.post(
            "/api/train",
            json={
                "source": "sklearn",
                "identifier": "iris",
                "target": "target",
                "model_name": "logistic_regression",
                "task": "classification",
                "params": {"max_iter": 200},
                "log_mlflow": True,
                "tracking_uri": uri,
                "experiment_name": "api_test_exp",
            },
        )
        assert res.status_code == 200, res.text
        run_id = res.json()["mlflow_run_id"]
        assert run_id
        assert res.json()["logged_to_mlflow"] is True

    def test_evaluate_logged_run(self, client, tmp_path):
        uri = f"sqlite:///{tmp_path}/eval.db"
        res = client.post(
            "/api/train",
            json={
                "source": "sklearn",
                "identifier": "iris",
                "target": "target",
                "model_name": "logistic_regression",
                "task": "classification",
                "params": {"max_iter": 200},
                "log_mlflow": True,
                "tracking_uri": uri,
                "experiment_name": "eval_test_exp",
            },
        )
        assert res.status_code == 200, res.text
        run_id = res.json()["mlflow_run_id"]
        assert run_id

        res = client.post(
            "/api/evaluate",
            json={
                "run_id": run_id,
                "source": "sklearn",
                "identifier": "iris",
                "target": "target",
                "tracking_uri": uri,
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["run_id"] == run_id
        assert data["metrics"]["accuracy"] > 0.85
        assert data["n_test"] > 0

    def test_run_detail_missing(self, client, tmp_path):
        import mlflow

        uri = f"sqlite:///{tmp_path}/missing.db"
        mlflow.set_tracking_uri(uri)
        res = client.get("/api/runs/deadbeef")
        assert res.status_code == 404
