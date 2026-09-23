import time

from fastapi.testclient import TestClient

from api.app import app
from api.service import column_stats, run_pipeline_snapshot, train_pipeline_snapshot


def client():
    with TestClient(app) as c:
        yield c


class TestParamSchemaMetadata:
    def test_group_and_advanced_in_models_api(self):
        with TestClient(app) as c:
            data = c.get("/api/models").json()
        lr = next(m for m in data if m["model_name"] == "logistic_regression")
        by_name = {p["name"]: p for p in lr["parameters"]}
        assert by_name["C"]["group"] == "regularization"
        assert by_name["C"]["description"]
        assert by_name["solver"]["advanced"] is True
        assert by_name["random_state"]["group"] == "reproducibility"

    def test_xgb_groups(self):
        with TestClient(app) as c:
            data = c.get("/api/models").json()
        xgb = next(m for m in data if m["model_name"] == "xgb_classifier")
        by_name = {p["name"]: p for p in xgb["parameters"]}
        assert by_name["learning_rate"]["group"] == "boosting"
        assert by_name["subsample"]["group"] == "sampling"
        assert by_name["subsample"]["advanced"] is True


class TestJobStages:
    def test_job_reports_pipeline_stages(self):
        with TestClient(app) as c:
            res = c.post(
                "/api/jobs/train",
                json={
                    "source": "sklearn",
                    "identifier": "iris",
                    "target": "target",
                    "model_name": "logistic_regression",
                    "task": "classification",
                    "params": {"max_iter": 300},
                },
            )
            assert res.status_code == 200
            job_id = res.json()["job_id"]
            status = None
            for _ in range(120):
                status = c.get(f"/api/jobs/{job_id}").json()
                if status["status"] in ("done", "error"):
                    break
                time.sleep(0.25)
            assert status is not None
            assert status["status"] == "done", status.get("error")
            assert status["stage"] == "done"
            assert status["pipeline"] == [
                "queued",
                "loading",
                "preparing",
                "training",
                "evaluating",
                "done",
            ]
            stage_names = [s["stage"] for s in status["stages"]]
            assert "loading" in stage_names
            assert "training" in stage_names
            assert stage_names[0] == "queued"
            assert status["result"]["pipeline"]
            assert status["result"]["pipeline"][0]["operation"] == "load_dataset"

    def test_job_with_preprocess_stage(self):
        with TestClient(app) as c:
            res = c.post(
                "/api/jobs/train",
                json={
                    "source": "sklearn",
                    "identifier": "iris",
                    "target": "target",
                    "model_name": "logistic_regression",
                    "task": "classification",
                    "params": {"max_iter": 200},
                    "operations": [{"operation": "shuffle", "random_state": 1}],
                },
            )
            assert res.status_code == 200
            job_id = res.json()["job_id"]
            status = None
            for _ in range(120):
                status = c.get(f"/api/jobs/{job_id}").json()
                if status["status"] in ("done", "error"):
                    break
                time.sleep(0.25)
            assert status is not None
            assert status["status"] == "done", status.get("error")
            assert "preprocessing" in status["pipeline"]
            assert status["result"]["preprocess_steps"]
            ops = [s["operation"] for s in status["result"]["pipeline"]]
            assert "shuffle" in ops


class TestTrainWithPreprocess:
    def test_training_applies_operations(self):
        with TestClient(app) as c:
            res = c.post(
                "/api/train",
                json={
                    "dataset": "iris",
                    "target": "target",
                    "model_name": "logistic_regression",
                    "task": "classification",
                    "params": {"max_iter": 300},
                    "operations": [
                        {"operation": "shuffle", "random_state": 7},
                        {"operation": "sample_rows", "n": 120, "random_state": 1},
                    ],
                },
            )
            assert res.status_code == 200, res.text
            data = res.json()
            assert data["n_train"] + data["n_val"] + data["n_test"] <= 120
            assert len(data["preprocess_steps"]) == 2
            assert data["preprocess_ops"]
            pipe_ops = [s["operation"] for s in data["pipeline"]]
            assert pipe_ops.index("shuffle") < pipe_ops.index("prepare_features")
            assert pipe_ops.index("sample_rows") < pipe_ops.index("train_model")


class TestMlflowMonitoring:
    def test_status_endpoint(self):
        with TestClient(app) as c:
            res = c.get("/api/mlflow/status")
            assert res.status_code == 200
            data = res.json()
            assert data["ui_url"]
            assert "tracking_uri" in data
            assert isinstance(data["server_up"], bool)
            assert data["experiment_name"] == "pytrain"


class TestDatasetStats:
    def test_preview_includes_column_stats(self):
        with TestClient(app) as c:
            res = c.post(
                "/api/datasets/preview",
                json={"source": "sklearn", "identifier": "iris", "n_rows": 5},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["columns_stats"]
            sepal = next(s for s in data["columns_stats"] if s["column"] == "sepal length (cm)")
            assert sepal["mean"] is not None
            assert sepal["missing"] == 0
            assert data["target"] == "target"
            assert data["class_distribution"]
            assert abs(sum(data["class_distribution"].values()) - 100.0) < 1.0
            assert data["duplicates"] >= 0

    def test_column_stats_numeric(self):
        import pandas as pd

        df = pd.DataFrame({"a": [1.0, 2.0, None], "b": ["x", "x", "y"]})
        stats = {s["column"]: s for s in column_stats(df)}
        assert stats["a"]["missing"] == 1
        assert stats["a"]["mean"] == 1.5
        assert stats["b"]["top"] == "x"
        assert stats["b"]["top_pct"] > 50


class TestEvaluateDetails:
    def test_evaluate_returns_confusion_matrix_and_samples(self, tmp_path):
        uri = f"sqlite:///{tmp_path}/eval_details.db"
        with TestClient(app) as c:
            res = c.post(
                "/api/train",
                json={
                    "source": "sklearn",
                    "identifier": "iris",
                    "target": "target",
                    "model_name": "logistic_regression",
                    "task": "classification",
                    "params": {"max_iter": 300},
                    "log_mlflow": True,
                    "tracking_uri": uri,
                    "experiment_name": "eval_details",
                },
            )
            assert res.status_code == 200, res.text
            run_id = res.json()["mlflow_run_id"]
            res = c.post(
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
            assert data["task"] == "classification"
            assert data["confusion_matrix"]["matrix"]
            labels = data["confusion_matrix"]["labels"]
            assert len(data["confusion_matrix"]["matrix"]) == len(labels)
            assert data["samples"]
            assert "actual" in data["samples"][0]
            assert "predicted" in data["samples"][0]

    def test_regression_samples(self, tmp_path):
        uri = f"sqlite:///{tmp_path}/eval_reg.db"
        with TestClient(app) as c:
            res = c.post(
                "/api/train",
                json={
                    "source": "sklearn",
                    "identifier": "diabetes",
                    "target": "target",
                    "model_name": "linear_regression",
                    "task": "regression",
                    "log_mlflow": True,
                    "tracking_uri": uri,
                    "experiment_name": "eval_reg",
                },
            )
            assert res.status_code == 200, res.text
            run_id = res.json()["mlflow_run_id"]
            res = c.post(
                "/api/evaluate",
                json={
                    "run_id": run_id,
                    "source": "sklearn",
                    "identifier": "diabetes",
                    "target": "target",
                    "tracking_uri": uri,
                },
            )
            assert res.status_code == 200, res.text
            data = res.json()
            assert data["task"] == "regression"
            assert data["confusion_matrix"] is None
            assert len(data["samples"]) > 10


class TestRunPipeline:
    def test_list_runs_includes_pipeline(self, tmp_path):
        uri = f"sqlite:///{tmp_path}/runs_pipe.db"
        with TestClient(app) as c:
            res = c.post(
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
                    "experiment_name": "pipe_exp",
                },
            )
            assert res.status_code == 200
            assert res.json()["pipeline"]

        import mlflow

        mlflow.set_tracking_uri(uri)
        from api.service import list_runs

        runs = list_runs(uri)
        assert runs
        ops = [s["operation"] for s in runs[0]["pipeline"]]
        assert "load_dataset" in ops
        assert "train_model" in ops
        assert all(s["status"] == "done" for s in runs[0]["pipeline"])

    def test_train_pipeline_snapshot_shape(self):
        payload = {
            "n_features": 4,
            "n_train": 100,
            "n_val": 10,
            "n_test": 20,
            "model_info": {
                "model_name": "logistic_regression",
                "task": "classification",
                "framework": "scikit-learn",
            },
            "mlflow_run_id": "abc123",
        }
        steps = train_pipeline_snapshot(
            payload,
            label="iris",
            target="target",
            task="classification",
            df_shape=(150, 5),
            logged=True,
            preprocess_steps=[
                {"operation": "shuffle", "args": {"random_state": 1}, "rows": 150, "cols": 5}
            ],
        )
        assert [s["operation"] for s in steps][:4] == [
            "load_dataset",
            "shuffle",
            "prepare_features",
            "split_data",
        ]
        assert any(s["operation"] == "log_mlflow" for s in steps)

    def test_run_pipeline_snapshot_params(self):
        steps = run_pipeline_snapshot(
            {
                "source": "sklearn",
                "identifier": "iris",
                "target": "target",
                "model_name": "logistic_regression",
                "task": "classification",
                "test_size": "0.2",
                "validation_size": "0.1",
                "mlflow_run_id": "deadbeef",
            }
        )
        assert any(s["operation"] == "log_mlflow" for s in steps)
