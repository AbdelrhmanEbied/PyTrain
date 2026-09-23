# PyTrain Usage Guide

A detailed walkthrough of the app: how to start it, what each tab does, how pipeline configs work, and how to use the HTTP API directly.

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and, for the frontend build, Node.

```bash
uv sync --group dev
uv run pytrain                 # http://127.0.0.1:8000
```

Options:

```bash
uv run pytrain --host 0.0.0.0 --port 9000
uv run pytrain --reload        # auto-reload during development
```

Equivalent without the console script:

```bash
uv run uvicorn api.app:app --reload
```

- UI: http://127.0.0.1:8000
- Interactive API docs: http://127.0.0.1:8000/docs
- Optional extra file readers (Excel, XML, SPSS): `uv sync --extra formats`
- Optional MLflow UI (also startable from the Monitoring tab): http://127.0.0.1:5000

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `PYTRAIN_HOST` | `127.0.0.1` | Bind host for `pytrain` |
| `PYTRAIN_PORT` | `8000` | Bind port for `pytrain` |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow tracking store |
| `MLFLOW_UI_PORT` | `5000` | Managed MLflow UI port |

## The seven tabs

### 1. Train

The main workflow.

1. **Dataset** — pick a source (`sklearn`, `local`, `huggingface`, or a registered name like `titanic`) and an identifier; set the target column (defaults are applied for known datasets).
2. **Model** — choose one of 32 models (28 sklearn + 4 XGBoost). Parameter forms are generated from the server-side schema; there is also a JSON escape hatch for advanced params.
3. **Split & options** — `test_size`, `validation_size`, `stratify`, `random_state`, task (`classification`/`regression`).
4. **Optional** — preprocessing operations (same catalog as the Preprocess tab), grid-search CV (`param_config`), MLflow logging (`log_mlflow`, `experiment_name`).
5. **Pipeline config (collapsible)** — load a YAML/JSON config (paste or file), **Validate**, **Validate file**, **Apply to form**, or **Validate & train**. See [Pipeline config](#pipeline-config).
6. **Train** — starts a background job. Stages stream live: `queued → loading → [preprocessing] → preparing → training → evaluating → [logging] → done`.
7. **Results** — train/val/test metrics, duration, pipeline/steps, MLflow run id (if logged). Buttons: download model, **Try this model** (opens Predict with this run).

### 2. Preprocess

Interactive **preview** of preprocessing (dry-run — nothing is saved to disk):

- Build a chain of operations: drop/rename columns, drop duplicates, drop/fill missing, convert dtype, parse dates, clean numeric, filter rows, shuffle, sample, remove/clip outliers, date features, numeric transforms.
- Each operation is validated against the server catalog (`PREPROCESSOR_METHODS`).
- Preview the result and inspect the operations audit log (rows/cols after each step).
- **To apply these when training:** put the same operations on the Train tab (or in a config file’s `preprocessing.operations`). At train time, those `operations` run first, then default feature preparation (`prepare_xy`) always runs.

### 3. Datasets

- Table of registered datasets (name, source, identifier, target) — uploads in `data/uploads/` are auto-registered.
- Preview any dataset by source/identifier without registering it.
- Upload a data file (CSV, JSON, Parquet, …) → becomes a registered local dataset.

### 4. Versions

Data versioning snapshots:

- **Create** a snapshot of a registered dataset → stored under `data/versions/{name}/{id}/` (parquet/csv) plus `index.json` metadata (rows, columns, dtypes, missing, duplicates, content hash).
- **Preview** a snapshot, **diff** two snapshots (row/col deltas, added/removed/changed columns), **restore** a snapshot as a local dataset, **delete** it.

### 5. Runs

MLflow run history (when `log_mlflow` is on):

- List runs; open a run for full detail (split config, hyperparameters, artifacts, metric history).
- Rename, soft-delete, restore, re-evaluate on the reproduced test split, download the model artifact as a zip.
- **Where are models?** — storage panel: project root, tracking DB, artifact root (`mlruns/`), uploads, data versions (also `GET /api/storage`).
- Run detail shows **Model location** (`artifact_uri`) for filesystem access.

### 6. Predict

Playground for models logged to MLflow:

1. Pick a run (id from the Runs tab or train results).
2. Fetch its feature schema (`GET /api/predict/schema`) — feature names, dtypes, task, target, classes.
3. Fill feature values and submit (`POST /api/predict`).
4. Get the class prediction (or regression value) plus per-class probabilities when available.

Missing required features → 400; unknown run / missing artifact → 404.

### 7. Monitoring

- MLflow tracking status: tracking URI, experiment run count, whether the UI is reachable.
- Start / stop the managed MLflow UI server (`mlflow server` on port 5000 by default). Logs append to `mlflow-ui.log`.
- Quick list of recent runs with links into the Runs view.

## Pipeline config

Configs are YAML or JSON. The root must be a mapping. Unknown keys are rejected.

Configs are **normalized early** to one canonical shape: the flat **`TrainRequest`** (what `POST /api/config/validate` returns as `request`). Nested `dataset` / `training` / `model` / `preprocessing` blocks are **input aliases only**.

**Precedence**

1. Canonical = flat `TrainRequest` fields after alias merge.
2. Aliases may repeat the same value; if the same field has **different** values in two places → error.
3. `dataset:` may be an object (`source`/`identifier`/…) or a registered name string (`dataset: titanic`).
4. `model:` may be a string id or `{ model_name, params, task, param_config }`.

**Preprocessing at train time**

1. `TrainRequest.operations` (from config or Train form) run first (if any).
2. Default feature prep (`prepare_xy`) **always** runs next (fill, encode, target handling).
3. The **Preprocess tab is a dry-run/preview** — it does not save changes; put the same operations on the Train form or in a config to apply them while training.

### Schema (basic)

| Key | Type | Description |
|---|---|---|
| `dataset` | object | Nested dataset ref: `source`, `identifier`, `split`, `dataset` (registered name), `target` |
| `source` | string | `sklearn` \| `local` \| `huggingface` (top-level alternative to `dataset.source`) |
| `identifier` | string | Dataset identifier (required if `source` is set) |
| `split` | string | HF split (optional) |
| `target` | string | Target column name |
| `training` | object | Nested training block (see below); flattened into the request |
| `preprocessing` | object | `{ operations: [...] }` or `{ ops: [...] }` |
| `operations` | list | Preprocessing operations (same list as `training.operations`) |
| `model` | string \| object | Model name, or `{ model_name, params, task, param_config }` |
| `model_name` | string | Model id (e.g. `logistic_regression`) — also allowed under `training` |
| `task` | string | `classification` \| `regression` |
| `params` | object | Hyperparameters (validated against the model schema) |
| `param_config` | object | Grid search: `{ grid: {...}, cv, scoring, refit, n_jobs, verbose }` |
| `test_size` | number | 0.05–0.5 (default 0.2) |
| `validation_size` | number \| null | 0.0–0.5 (default 0.1) |
| `stratify` | bool | Stratified split (forced off for regression) |
| `random_state` | int | Seed (default 42) |
| `log_mlflow` | bool | Log the run to MLflow |
| `experiment_name` | string | MLflow experiment (default `pytrain`) |
| `tracking_uri` | string | Override tracking URI |

**`training` block keys:** `model_name`, `task`, `params`, `param_config`, `test_size`, `validation_size`, `stratify`, `random_state`, `log_mlflow`, `experiment_name`, `tracking_uri`, `target`, `source`, `identifier`, `split`, `dataset`, `operations`.

**`dataset` block keys:** `source`, `identifier`, `split`, `dataset`, `target`.

**Preprocessing operations** (each item needs `operation` plus method args):

`drop_columns`, `rename_columns`, `drop_duplicates`, `drop_missing_rows`, `fill_missing`, `convert_dtype`, `parse_dates`, `clean_numeric`, `filter_rows`, `shuffle`, `sample_rows`, `remove_outliers`, `clip_outliers`, `create_date_features`, `transform_numeric`.

If neither `dataset` nor `identifier` is set, the config defaults to the registered dataset `titanic`.

### Example: YAML

Ready-made examples live in `docs/examples/` (`train-iris.yaml`, `train-iris.json`, `train-titanic-grid.yaml`).

```yaml
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
validation_size: 0.1
stratify: true
random_state: 42

log_mlflow: true
experiment_name: iris_rf

preprocessing:
  operations:
    - operation: drop_missing_rows
      how: any
```

### Example: JSON

```json
{
  "dataset": {
    "source": "huggingface",
    "identifier": "phihung/titanic",
    "target": "Survived"
  },
  "training": {
    "model_name": "logistic_regression",
    "task": "classification",
    "params": { "C": 1.0, "max_iter": 500 },
    "test_size": 0.2,
    "stratify": true,
    "random_state": 42,
    "log_mlflow": true,
    "experiment_name": "titanic_lr"
  },
  "param_config": {
    "grid": { "C": [0.1, 1.0, 10.0] },
    "cv": 3,
    "scoring": "accuracy"
  }
}
```

### Example: minimal (registered dataset name)

```yaml
dataset: titanic
target: Survived
model: random_forest_classifier
task: classification
params:
  n_estimators: 100
log_mlflow: true
```

### Using configs

- **UI**: Train tab → expandable **Load config (YAML / JSON)** panel → paste or pick a file → Validate / Apply / Validate & train.
- **API**:
  - `POST /api/config/validate` — body `{ "content": "...", "format": "yaml"|"json"|null }` → normalized request + YAML/JSON previews
  - `POST /api/config/validate-file` — multipart file upload (format inferred from extension if omitted)
  - `POST /api/config/to-request` — normalized `TrainRequest` JSON
  - `POST /api/config/run` — validate, start background job → `{ job_id, ... }`
  - `POST /api/config/run-sync` — validate and train synchronously

## HTTP API quick recipes

Full table is in the [README](../README.md#api-overview). Common flows:

### Train a model (sync)

```bash
curl -s -X POST http://127.0.0.1:8000/api/train \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset": "iris",
    "target": "target",
    "model_name": "logistic_regression",
    "task": "classification",
    "params": {"max_iter": 300},
    "log_mlflow": true,
    "experiment_name": "curl_exp"
  }'
```

### Train as a background job

```bash
# start
curl -s -X POST http://127.0.0.1:8000/api/jobs/train \
  -H 'Content-Type: application/json' \
  -d '{"dataset": "iris", "target": "target", "model_name": "random_forest_classifier"}'
# → {"job_id": "...", "status": "running"}

# poll
curl -s http://127.0.0.1:8000/api/jobs/<job_id>
```

Job `stage` progresses through `queued → loading → … → done`; `result` is populated when `status` is `done`.

Jobs persist in SQLite (`data/jobs.db`, last 200 kept). Finished job history survives a server restart; a job still `running` at shutdown is marked `error` (“Interrupted by server restart”).

### Validate a config file

```bash
curl -s -X POST http://127.0.0.1:8000/api/config/validate-file \
  -F "file=@my-train.yaml"
```

### List models and their parameter schemas

```bash
curl -s http://127.0.0.1:8000/api/models
```

### Snapshot a dataset

```bash
curl -s -X POST http://127.0.0.1:8000/api/data-versions \
  -H 'Content-Type: application/json' \
  -d '{"dataset": "iris", "message": "baseline"}'
```

### Predict with a logged run

```bash
RUN=<mlflow_run_id>
curl -s "http://127.0.0.1:8000/api/predict/schema?run_id=$RUN"
curl -s -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"run_id": "'"$RUN"'", "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}}'
```

### Storage paths

```bash
curl -s http://127.0.0.1:8000/api/storage
```

### Start / stop the MLflow UI

```bash
curl -s -X POST http://127.0.0.1:8000/api/mlflow/ui/start
curl -s -X POST http://127.0.0.1:8000/api/mlflow/ui/stop
```

## Where things live on disk

| Path | Contents |
|---|---|
| `mlflow.db` | MLflow tracking DB (sqlite, gitignored) |
| `mlruns/` | MLflow run artifacts / model registry (gitignored) |
| `data/jobs.db` | Background job history (sqlite, gitignored) |
| `data/uploads/` | Uploaded dataset files (gitignored) |
| `data/versions/` | Dataset snapshots + `index.json` |
| `data/test_datasets/` | Builtin local test data (large files gitignored) |
| `static/` | Built frontend assets (generated by Vite) |
| `mlflow-ui.log` | Managed MLflow UI server log |

Also available live via `GET /api/storage`.

## Models

32 models total — query `GET /api/models` for names, task, and full parameter schemas (types, defaults, min/max, choices).

- **Classification (sklearn)**: `logistic_regression`, `random_forest_classifier`, `extra_trees_classifier`, `gradient_boosting_classifier`, `hist_gradient_boosting_classifier`, `adaboost_classifier`, `bagging_classifier`, `decision_tree_classifier`, `knn_classifier`, `svm_classifier`, `gaussian_nb_classifier`, `sgd_classifier`, `mlp_classifier`
- **Regression (sklearn)**: `linear_regression`, `random_forest_regressor`, `extra_trees_regressor`, `gradient_boosting_regressor`, `hist_gradient_boosting_regressor`, `adaboost_regressor`, `bagging_regressor`, `decision_tree_regressor`, `knn_regressor`, `svm_regressor`, `sgd_regressor`, `mlp_regressor`, `ridge_regression`, `lasso_regression`, `elasticnet_regression`
- **XGBoost**: `xgb_classifier`, `xgb_rf_classifier`, `xgb_regressor`, `xgb_rf_regressor`

## Development

```bash
uv run ruff check .        # lint
uv run ruff format .       # format
uv run pytest              # tests (325)

cd frontend
npm install
npm run build              # production build → ../static
npm run dev                # dev server (proxies /api to :8000)
```

Do not run `npm run build` in parallel with pytest — Vite's `emptyOutDir` clears `static/` and will break index-serving tests.

See [Architecture.md](./Architecture.md) for the full architecture document.
