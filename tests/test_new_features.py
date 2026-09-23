import json

import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


VALID_YAML = """
dataset:
  dataset: iris
  target: target
training:
  model_name: logistic_regression
  task: classification
  params:
    max_iter: 300
  test_size: 0.2
  validation_size: 0.1
preprocessing:
  operations:
    - operation: shuffle
      random_state: 42
"""

INVALID_YAML = """
dataset:
  dataset: iris
training:
  model_name: nope
  task: classification
"""

VALID_JSON = json.dumps(
    {
        "dataset": {"dataset": "iris", "target": "target"},
        "training": {
            "model_name": "logistic_regression",
            "task": "classification",
            "params": {"max_iter": 200},
        },
    }
)


class TestConfigValidate:
    def test_valid_yaml(self, client):
        res = client.post("/api/config/validate", json={"content": VALID_YAML, "format": "yaml"})
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["valid"] is True
        assert data["request"]["model_name"] == "logistic_regression"
        assert data["request"]["dataset"] == "iris"
        assert data["request"]["operations"][0]["operation"] == "shuffle"
        assert "model_name" in data["normalized_yaml"]
        assert data["normalized_json"]

    def test_valid_json(self, client):
        res = client.post("/api/config/validate", json={"content": VALID_JSON, "format": "json"})
        assert res.status_code == 200, res.text
        assert res.json()["valid"] is True

    def test_invalid_model(self, client):
        res = client.post("/api/config/validate", json={"content": INVALID_YAML, "format": "yaml"})
        assert res.status_code == 400

    def test_invalid_yaml_syntax(self, client):
        res = client.post("/api/config/validate", json={"content": "foo: [bar", "format": "yaml"})
        assert res.status_code == 400

    def test_unknown_key(self, client):
        res = client.post(
            "/api/config/validate",
            json={"content": "nope: 1\n", "format": "yaml"},
        )
        assert res.status_code == 400

    def test_validate_file(self, client):
        res = client.post(
            "/api/config/validate-file",
            files={"file": ("pipe.yaml", VALID_YAML.encode(), "application/yaml")},
        )
        assert res.status_code == 200, res.text
        assert res.json()["valid"] is True

    def test_to_request(self, client):
        res = client.post("/api/config/to-request", json={"content": VALID_YAML, "format": "yaml"})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["model_name"] == "logistic_regression"
        assert body["dataset"] == "iris"

    def test_config_run_job(self, client):
        res = client.post("/api/config/run", json={"content": VALID_YAML, "format": "yaml"})
        assert res.status_code == 200, res.text
        job_id = res.json()["job_id"]
        for _ in range(120):
            status = client.get(f"/api/jobs/{job_id}").json()
            if status["status"] in ("done", "error"):
                break
            import time

            time.sleep(0.25)
        assert status["status"] == "done", status.get("error")

    def test_config_run_sync(self, client):
        res = client.post("/api/config/run-sync", json={"content": VALID_YAML, "format": "yaml"})
        assert res.status_code == 200, res.text
        assert res.json()["model_info"]["model_name"] == "logistic_regression"


class TestDataVersions:
    def test_list_empty(self, client, monkeypatch, tmp_path):
        import api.data_versions as dv

        monkeypatch.setattr(dv, "VERSIONS_DIR", tmp_path / "versions")
        res = client.get("/api/data-versions")
        assert res.status_code == 200
        assert res.json() == []

    def test_create_list_get_delete(self, client, monkeypatch, tmp_path):
        import api.data_versions as dv

        monkeypatch.setattr(dv, "VERSIONS_DIR", tmp_path / "versions")
        res = client.post(
            "/api/data-versions",
            json={"dataset": "iris", "message": "first snapshot"},
        )
        assert res.status_code == 200, res.text
        created = res.json()
        assert created["dataset"] == "iris"
        assert created["rows"] == 150
        assert created["hash"]
        vid = created["id"]

        res = client.get("/api/data-versions")
        assert res.status_code == 200
        assert any(v["id"] == vid for v in res.json())

        res = client.get(f"/api/data-versions/{vid}")
        assert res.status_code == 200
        assert res.json()["id"] == vid

        res = client.get(f"/api/data-versions/{vid}/preview?n_rows=5")
        assert res.status_code == 200
        prev = res.json()
        assert len(prev["preview"]) == 5

        res = client.delete(f"/api/data-versions/{vid}")
        assert res.status_code == 200
        assert res.json()["deleted"] is True

        res = client.get(f"/api/data-versions/{vid}")
        assert res.status_code == 404

    def test_diff_and_restore(self, client, monkeypatch, tmp_path):
        import api.data_versions as dv

        monkeypatch.setattr(dv, "VERSIONS_DIR", tmp_path / "versions")
        a = client.post("/api/data-versions", json={"dataset": "iris", "message": "a"}).json()
        b = client.post("/api/data-versions", json={"dataset": "iris", "message": "b"}).json()

        res = client.get(f"/api/data-versions/diff?a={a['id']}&b={b['id']}")
        assert res.status_code == 200, res.text
        diff = res.json()
        assert diff["same_hash"] is True
        assert diff["row_delta"] == 0

        res = client.post(f"/api/data-versions/{a['id']}/restore", json={})
        assert res.status_code == 200, res.text
        restored = res.json()
        assert restored["rows"] == 150
        assert restored["from_version"] == a["id"]


class TestMlflowUiRoutes:
    def test_status_endpoint(self, client):
        res = client.get("/api/mlflow/ui/status")
        assert res.status_code == 200
        data = res.json()
        assert "ui_url" in data
        assert "server_up" in data

    def test_stop_when_not_running(self, client):
        res = client.post("/api/mlflow/ui/stop")
        assert res.status_code == 200
        data = res.json()
        assert data["stopped"] is False or data["was_running"] is False


class TestRunManagement:
    def _train_logged(self, client, tmp_path):
        uri = f"sqlite:///{tmp_path}/mgmt.db"
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
                "experiment_name": "mgmt_exp",
            },
        )
        assert res.status_code == 200, res.text
        return res.json()["mlflow_run_id"], uri

    def test_run_details(self, client, tmp_path):
        run_id, uri = self._train_logged(client, tmp_path)
        res = client.get(f"/api/runs/{run_id}/details", params={"tracking_uri": uri})
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["run_id"] == run_id
        assert "duration_ms" in data
        assert "metric_history" in data
        assert "split_config" in data
        assert "dataset_info" in data

    def test_rename_delete_restore(self, client, tmp_path):
        run_id, uri = self._train_logged(client, tmp_path)
        q = {"tracking_uri": uri}

        res = client.patch(f"/api/runs/{run_id}", json={"name": "my experiment"}, params=q)
        assert res.status_code == 200, res.text
        assert res.json()["name"] == "my experiment"

        res = client.delete(f"/api/runs/{run_id}", params=q)
        assert res.status_code == 200
        assert res.json()["deleted"] is True

        res = client.post(f"/api/runs/{run_id}/restore", params=q)
        assert res.status_code == 200
        assert res.json()["restored"] is True


class TestExpandedModels:
    def test_api_lists_new_models(self, client):
        names = [m["model_name"] for m in client.get("/api/models").json()]
        for expected in (
            "extra_trees_classifier",
            "hist_gradient_boosting_classifier",
            "ridge_regression",
            "elasticnet_regression",
            "knn_classifier",
            "svm_classifier",
            "gaussian_nb_classifier",
            "mlp_classifier",
            "adaboost_regressor",
            "bagging_classifier",
            "decision_tree_regressor",
            "xgb_rf_classifier",
            "xgb_rf_regressor",
        ):
            assert expected in names

    def test_train_extra_trees(self, client):
        res = client.post(
            "/api/train",
            json={
                "dataset": "iris",
                "target": "target",
                "model_name": "extra_trees_classifier",
                "task": "classification",
                "params": {"n_estimators": 30, "random_state": 42},
            },
        )
        assert res.status_code == 200, res.text
        assert res.json()["test_metrics"]["accuracy"] > 0.85

    def test_train_ridge(self, client):
        res = client.post(
            "/api/train",
            json={
                "dataset": "diabetes",
                "target": "target",
                "model_name": "ridge_regression",
                "task": "regression",
                "params": {"alpha": 1.0},
            },
        )
        assert res.status_code == 200, res.text
        assert "r2" in res.json()["test_metrics"]


class TestStorage:
    def test_storage_paths(self, client):
        res = client.get("/api/storage")
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["project_root"]
        assert data["tracking_uri"]
        assert data["artifact_root"]
        assert data["data_uploads"]
        assert data["data_versions"]
        assert data["tracking_db"] is None or data["tracking_db"].endswith(".db")

    def test_storage_custom_uri(self, client, tmp_path):
        uri = f"sqlite:///{tmp_path}/store.db"
        res = client.get("/api/storage", params={"tracking_uri": uri})
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["tracking_uri"] == uri
        assert data["tracking_db"] == str((tmp_path / "store.db").resolve())


class TestPredict:
    def _train_logged(self, client, tmp_path):
        uri = f"sqlite:///{tmp_path}/predict.db"
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
                "experiment_name": "predict_exp",
            },
        )
        assert res.status_code == 200, res.text
        return res.json()["mlflow_run_id"], uri

    def test_schema_and_predict(self, client, tmp_path):
        run_id, uri = self._train_logged(client, tmp_path)

        res = client.get(
            "/api/predict/schema",
            params={"run_id": run_id, "tracking_uri": uri},
        )
        assert res.status_code == 200, res.text
        schema = res.json()
        assert schema["run_id"] == run_id
        assert schema["task"] == "classification"
        assert schema["n_features"] > 0
        assert schema["features"][0]["name"]
        assert "sample" in schema["features"][0]
        assert schema["classes"] is not None

        features = {f["name"]: f["sample"] for f in schema["features"]}
        res = client.post(
            "/api/predict",
            json={"run_id": run_id, "features": features, "tracking_uri": uri},
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["run_id"] == run_id
        assert data["prediction"] in schema["classes"]
        assert data["probabilities"] is not None
        assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-5

    def test_predict_missing_feature(self, client, tmp_path):
        run_id, uri = self._train_logged(client, tmp_path)
        res = client.post(
            "/api/predict",
            json={"run_id": run_id, "features": {"nope": 1}, "tracking_uri": uri},
        )
        assert res.status_code == 400

    def test_schema_unknown_run(self, client, tmp_path):
        uri = f"sqlite:///{tmp_path}/empty.db"
        res = client.get(
            "/api/predict/schema",
            params={"run_id": "deadbeef", "tracking_uri": uri},
        )
        assert res.status_code in (400, 404)
