# PyTrain

**No-code machine learning for beginners** — load a dataset, pick a model, click Train, and track your experiments — all from the browser. No Python required.

PyTrain is built for people who are **new to ML** or **don’t write code day-to-day**: guided forms, sensible defaults, plain-language labels, and a full experiment history you can browse, compare, and reuse.

> **v1 scope:** tabular models only — **scikit-learn** and **XGBoost** (classification & regression).  
> **Coming soon:** deep learning, LLM fine-tuning, and deeper Hugging Face integrations (separate release).

## Features

- **Dataset sources** — pick a source and type an identifier:
  - `sklearn` — builtin datasets (`iris`, `wine`, `breast_cancer`, `diabetes`, `digits`, `linnerud`)
  - `local` — path to a file on disk (CSV, JSON, Parquet, Excel, Feather, XML, SAS, SPSS, Stata)
  - `huggingface` — any HF dataset identifier (downloads on demand), with optional split
  - **Upload** — send a file through the UI and it becomes a registered local dataset
- **Preprocessing** — chainable operations (drop/rename/fill/filter/outliers/transforms/date features) with a live preview and an operations audit log (preview-only; apply the same ops when you train)
- **Training** — scikit-learn and XGBoost models, schema-validated hyperparameters, train/val/test splits, optional grid-search CV, optional preprocessing ops applied during training
- **Pipeline config** — expandable YAML/JSON panel inside the Train tab; validate before run, apply the normalized request to the form, or start training from it (power-user / share-a-recipe feature)
- **Data versioning** — snapshot datasets to versioned parquet/csv files, preview, diff, restore, and delete
- **Async jobs** — `POST /api/jobs/train` runs training in the background; poll `GET /api/jobs/{id}`; stages are persisted in SQLite (`data/jobs.db`) so history survives a restart (in-flight jobs are marked interrupted)
- **MLflow (Experiment history)** — optional run logging to a local sqlite tracking store (`mlflow.db`); start/stop the MLflow UI from the Monitoring tab. In the product UI this is framed as **Experiments / Run history**, not “MLflow”
- **Run management** — list past runs, full single-run details (split, hyperparams, artifacts, metric history), rename, restore, delete, re-evaluate, download the model artifact as a zip
- **Storage paths** — one-click view of where tracking DB, model artifacts (`mlruns`), uploads, and data versions live on disk
- **Predict playground** — pick a logged run, fill its feature schema, and get class predictions + probabilities
- **Model registry** — 32 models (28 sklearn + 4 XGBoost) with parameter schemas for form/JSON editing and config validation

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
uv run pytrain
```

`pytrain` is the console entry (`api/cli.py`); it starts the server on http://127.0.0.1:8000. Options and env vars:

```bash
uv run pytrain --host 0.0.0.0 --port 9000
uv run pytrain --reload
# or: PYTRAIN_HOST / PYTRAIN_PORT
```

Or run the server directly:

```bash
uv run uvicorn api.app:app --reload
```

Open http://127.0.0.1:8000 — interactive API docs are at http://127.0.0.1:8000/docs.

Optional file-format readers (Excel, XML, SPSS):

```bash
uv sync --extra formats
```

Optional MLflow UI (also startable from the Monitoring tab):

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

## Pipeline config (YAML / JSON)

Configs validate into a canonical **`TrainRequest`** and can be used from the Train tab’s config panel or the `/api/config/*` endpoints. Nested `dataset` / `training` / `model` / `preprocessing` blocks are **input aliases only** — they are normalized early; conflicting values for the same field in two places are rejected.

**Precedence / rules**

1. Canonical shape is the flat `TrainRequest` (what `/api/config/validate` returns as `request`).
2. Aliases: `dataset: {...}` or `dataset: "name"`, `training: {...}`, `model: "id" | {…}`, `preprocessing: {operations|ops}`, plus top-level keys.
3. If the same field appears in two alias locations with **different** values → error. Same value twice is fine.
4. **Preprocessing at train time:** `TrainRequest.operations` (from config / Train form) run first, then default feature preparation (`prepare_xy`) always runs (fill, encode, target handling). The Preprocess tab is a **dry-run / preview** and does not persist changes.

Basic schema — root keys:

| Key | Type | Description |
|---|---|---|
| `dataset` | object \| string | `{ source, identifier, split, dataset, target }`, or `"registered-name"` shorthand |
| `source` / `identifier` / `split` / `target` | — | Top-level dataset ref alternative |
| `training` | object | Flattened training settings (same keys as below) |
| `model` | string \| object | Model id, or `{ model_name, params, task, param_config }` |
| `model_name` | string | e.g. `logistic_regression` (32 models; see `GET /api/models`) |
| `task` | string | `classification` \| `regression` |
| `params` | object | Hyperparameters (schema-validated) |
| `param_config` | object | Grid search: `{ grid, cv, scoring, refit, n_jobs, verbose }` |
| `preprocessing` | object | `{ operations: [...] }` — chainable preprocessing ops |
| `test_size` | 0.05–0.5 | Test split fraction (default 0.2) |
| `validation_size` | 0.0–0.5 \| null | Validation split (default 0.1) |
| `stratify` | bool | Stratified split (off for regression) |
| `random_state` | int | Seed (default 42) |
| `log_mlflow` | bool | Log run to experiment history |
| `experiment_name` / `tracking_uri` | string | Experiment name / store override |

Example training configs (also under `docs/examples/`) — YAML:

```yaml
# docs/examples/train-iris.yaml
dataset:
  source: sklearn
  identifier: iris
  target: target
model:
  model_name: random_forest_classifier
  task: classification
  params:
    n_estimators: 200
    max_depth: 8
    random_state: 42
test_size: 0.2
stratify: true
log_mlflow: true
experiment_name: iris_rf
preprocessing:
  operations:
    - operation: drop_missing_rows
      how: any
```

Equivalent JSON (`docs/examples/train-iris.json`):

```json
{
  "dataset": { "source": "sklearn", "identifier": "iris", "target": "target" },
  "training": {
    "model_name": "random_forest_classifier",
    "task": "classification",
    "params": { "n_estimators": 200, "max_depth": 8, "random_state": 42 },
    "test_size": 0.2,
    "stratify": true,
    "log_mlflow": true,
    "experiment_name": "iris_rf"
  },
  "preprocessing": {
    "operations": [{ "operation": "drop_missing_rows", "how": "any" }]
  }
}
```

Validate / run:

```bash
curl -s -X POST http://127.0.0.1:8000/api/config/validate \
  -H 'Content-Type: application/json' \
  -d '{"content": "...", "format": "yaml"}'
curl -s -X POST http://127.0.0.1:8000/api/config/validate-file -F "file=@train.yaml"
curl -s -X POST http://127.0.0.1:8000/api/config/run -H 'Content-Type: application/json' \
  -d '{"content": "...", "format": "json"}'
```

Full guide: [docs/USAGE.md](docs/USAGE.md).

## API overview

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/models` | Registered models + parameter schemas |
| GET | `/api/datasets` | Registered datasets (name, source, identifier, target) |
| POST | `/api/datasets/preview` | Preview any dataset by source/identifier |
| POST | `/api/datasets/upload` | Upload a data file |
| GET | `/api/datasets/{name}` | Preview a registered dataset by name |
| POST | `/api/preprocess` | Apply preprocessing operations (preview / dry-run), return preview + log |
| POST | `/api/train` | Train synchronously (optional `operations` applied first) |
| POST | `/api/jobs/train` | Start a background training job |
| GET | `/api/jobs/{job_id}` | Job status / result (stages, pipeline) |
| POST | `/api/config/validate` | Validate YAML/JSON pipeline config |
| POST | `/api/config/validate-file` | Validate an uploaded YAML/JSON file |
| POST | `/api/config/to-request` | Convert config to a TrainRequest |
| POST | `/api/config/run` | Validate then start a training job |
| POST | `/api/config/run-sync` | Validate then train synchronously |
| GET | `/api/data-versions` | List data snapshots |
| POST | `/api/data-versions` | Create a snapshot |
| GET | `/api/data-versions/diff` | Diff two snapshots |
| GET | `/api/data-versions/{id}` | Snapshot metadata |
| GET | `/api/data-versions/{id}/preview` | Preview a snapshot |
| POST | `/api/data-versions/{id}/restore` | Restore a snapshot to a local dataset |
| DELETE | `/api/data-versions/{id}` | Delete a snapshot |
| GET | `/api/runs` | Experiment run history |
| GET | `/api/runs/{run_id}` | Run detail |
| GET | `/api/runs/{run_id}/details` | Full run details (duration, artifacts, metric history) |
| PATCH | `/api/runs/{run_id}` | Rename a run |
| DELETE | `/api/runs/{run_id}` | Delete a run |
| POST | `/api/runs/{run_id}/restore` | Restore a deleted run |
| GET | `/api/runs/{run_id}/model` | Download model artifact (zip) |
| POST | `/api/evaluate` | Evaluate a logged run on its test split |
| GET | `/api/storage` | Where tracking DB, artifacts, uploads, and versions live |
| GET | `/api/predict/schema` | Feature schema for a logged run |
| POST | `/api/predict` | Predict with a logged run’s model |
| GET | `/api/mlflow/status` | Tracking URI, UI URL, server reachability, run count |
| GET | `/api/mlflow/ui/status` | Managed MLflow UI process status |
| POST | `/api/mlflow/ui/start` | Start the MLflow UI server |
| POST | `/api/mlflow/ui/stop` | Stop the MLflow UI server |

## Development

```bash
uv run ruff check .        # lint
uv run ruff format .       # format
uv run pytest              # tests
```

Frontend (React + Vite; builds into `static/`):

```bash
cd frontend
npm install
npm run build              # production build → ../static
npm run dev                # dev server (proxies /api to :8000)
```

CI runs lint + tests on every push and pull request (`.github/workflows/ci.yml`).

## Project layout

```
api/                 FastAPI app, routers, schemas, service package, job store, CLI
  service/           Orchestration split: datasets, preprocess, models, training, runs, predict, storage
data/                Dataset loading and preprocessing library
models/              Model abstractions and parameter schemas (sklearn, xgboost)
training/            Trainers, metrics, grid search, results
mlflow_integration/  Experiment tracking wrapper (MLflow)
frontend/            React + Vite source (npm run build → static/)
static/              Built frontend assets (generated)
docs/                Architecture + usage guide + example configs
tests/               Pytest suite
```

- [docs/Architecture.md](docs/Architecture.md) — full architecture document
- [docs/USAGE.md](docs/USAGE.md) — detailed usage guide (tabs, configs, API recipes)

## License

Licensed under the [Apache License 2.0](LICENSE).

## Roadmap

- **Now (v1):** no-code tabular training — scikit-learn + XGBoost, experiment history, versions, predict
- **Next:** deeper guidance/defaults for beginners, more gradient-boosting options
- **Later release:** deep learning, LLM fine-tuning, richer Hugging Face integration
