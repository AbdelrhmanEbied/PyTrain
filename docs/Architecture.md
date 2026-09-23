# PyTrain Architecture

## Overview

PyTrain is a **no-code ML playground for beginners**: load datasets, preprocess, train, evaluate, and track experiments — through a FastAPI backend and a React SPA frontend. v1 supports **tabular models only (scikit-learn + XGBoost)**; deep learning and LLM fine-tuning are planned for a later release. Training is CPU-bound Python run in worker threads so the event loop stays responsive; long trainings run as background jobs persisted in SQLite.

```
Browser (static/ SPA, React + Vite)
   │  fetch JSON / multipart
   ▼
FastAPI (api/)
   ├─ routes/            HTTP layer, all async
   ├─ service/           orchestration package (sync, executed via threadpool)
   │    ├─ datasets.py   registry, load, preview, upload
   │    ├─ preprocess.py ops + prepare_xy (train-time feature prep)
   │    ├─ models.py     model/trainer factories + schema listing
   │    ├─ training.py   split, run_training, pipeline snapshots
   │    ├─ runs.py       MLflow run history, evaluate, status
   │    ├─ predict.py    schema + predict for logged runs
   │    └─ storage.py    on-disk path report
   ├─ schemas.py         Pydantic contracts
   ├─ pipeline_config.py YAML/JSON config → TrainRequest (alias normalize)
   ├─ data_versions.py   dataset snapshots (parquet/csv + index)
   ├─ mlflow_ui.py       managed MLflow UI subprocess
   ├─ jobs.py            SQLite-backed background job store (data/jobs.db)
   ├─ cli.py             `pytrain` console entry (uvicorn)
   └─ errors.py          exception → HTTP status mapping
   │
   ▼
Domain libraries
   ├─ data/                 DataLoader (local | sklearn | huggingface)
   │                         Preprocessor (chainable, op-logged)
   ├─ models/               BaseModel → EstimatorModel → Sklearn/XGBoost
   │                         ParameterSchema validation (32 models)
   ├─ training/             BaseTrainer → EstimatorTrainer (fit/evaluate/GridSearchCV)
   └─ mlflow_integration/   MLflowTracker (params/metrics/model logging)
```

## Backend

### Layering

1. **HTTP layer** (`api/app.py`, `api/routes/*`) — `async def` handlers only. Blocking work (data loading, training, MLflow I/O, file I/O) is dispatched with `starlette.concurrency.run_in_threadpool`. No business logic beyond request/response mapping.
2. **Service layer** (`api/service/`) — plain synchronous modules composing the domain libraries: `datasets`, `preprocess`, `models`, `training`, `runs`, `predict`, `storage`. Public names are re-exported from `api.service` so routes keep a single import surface.
3. **Domain layer** (`data/`, `models/`, `training/`, `mlflow_integration/`) — framework-aware but FastAPI-unaware; fully unit-tested.

### App factory

`api/app.py::create_app()`:

- registers exception handlers (`api/errors.py`): `FileNotFoundError → 404`, `ModelError|ValueError|KeyError|TypeError|AttributeError → 400`
- includes routers in order: `training`, `runs`, `datasets`, `models`, `monitoring`, `config`, `data_versions` (static-literal routes like `/datasets/preview` are declared before the dynamic `/datasets/{name}`)
- adds permissive CORS
- mounts `/static` and serves the SPA shell at `/`
- `GET /api/health`

### Routers

| Router | File | Endpoints |
|---|---|---|
| training | `api/routes/training.py` | `POST /api/train`, `POST /api/jobs/train`, `GET /api/jobs/{id}` |
| runs | `api/routes/runs.py` | `GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/details`, `PATCH /api/runs/{id}`, `DELETE /api/runs/{id}`, `POST /api/runs/{id}/restore`, `GET /api/runs/{id}/model`, `POST /api/evaluate`, `GET /api/storage`, `GET /api/predict/schema`, `POST /api/predict` |
| datasets | `api/routes/datasets.py` | `GET /api/datasets`, `POST /api/datasets/preview`, `POST /api/datasets/upload`, `POST /api/preprocess`, `GET /api/datasets/{name}` |
| models | `api/routes/models.py` | `GET /api/models` |
| monitoring | `api/routes/monitoring.py` | `GET /api/mlflow/status`, `GET /api/mlflow/ui/status`, `POST /api/mlflow/ui/start`, `POST /api/mlflow/ui/stop` |
| config | `api/routes/config.py` | `POST /api/config/validate`, `POST /api/config/validate-file`, `POST /api/config/to-request`, `POST /api/config/run`, `POST /api/config/run-sync` |
| data versions | `api/routes/data_versions.py` | `GET/POST /api/data-versions`, `GET /api/data-versions/diff`, `GET /api/data-versions/{id}`, `GET .../{id}/preview`, `POST .../{id}/restore`, `DELETE .../{id}` |

`POST /api/preprocess` returns a `steps` array (operation, args, rows/cols after each op) that the frontend renders as a pipeline graph.

### Dataset addressing model

Every dataset reference is `(source, identifier, split?)`:

| Source | Identifier | Behavior |
|---|---|---|
| `sklearn` | dataset name | loads a builtin sklearn dataset |
| `local` | file path | reads a local file (CSV, JSON, Parquet, Excel, Feather, XML, SAS, SPSS, Stata) |
| `huggingface` | HF repo id | `datasets.load_dataset(...)`, optional `split` |

A registered name (e.g. `titanic`, `iris`, uploads) can also be used; `resolve_ref()` maps a name to its source/identifier pair. Uploads land in `data/uploads/` and are auto-registered by scanning that directory.

Requests may use either form:

```json
{"source": "huggingface", "identifier": "openai/gsm8k", "split": "train"}
{"dataset": "titanic"}
```

### Training flow

```
TrainRequest
  → resolve_ref → load_frame (threadpool)
  → [optional] Preprocessor ops from request.operations
  → prepare_xy (target split, default cleaning, one-hot, encode)
  → split_arrays (Preprocessor.split: train/val/test, optional stratify)
  → make_model (registry + param validation) → make_trainer
  → [optional] ParamConfig → GridSearchCV
  → fit → train metrics → val metrics → test metrics
  → [optional] MLflowTracker: params (incl. dataset provenance) + metrics + model
  → TrainResponse
```

Stratification is forced off for regression tasks. Job stages: `queued → loading → [preprocessing] → preparing → training → evaluating → [logging] → done` (or `error`).

### Pipeline config

`api/pipeline_config.py` parses YAML/JSON into a validated, canonical `TrainRequest`:

- `parse_config_text(text, fmt)` — auto-detects JSON (leading `{`) vs YAML
- `config_to_train_request(data)` — accepts input **aliases** only (`dataset` object or name string, `training`, `model` string/object, `preprocessing`, top-level keys) and **normalizes early** to flat `TrainRequest` fields; conflicting values for the same field in two places raise; model/params validated against the live registry; operations against `PREPROCESSOR_METHODS`
- `validate_config_text` returns the normalized request as both YAML and JSON

Config is used by the Train tab's expandable panel and by the `/api/config/*` endpoints (validate / validate-file / to-request / async run / sync run).

### Preprocessing precedence

One documented order at train time:

1. **`TrainRequest.operations`** (optional) — user ops from the Train form or config, applied first via `apply_operations`.
2. **`prepare_xy`** — always runs next: target column handling, titanic drops when relevant, fill missing, one-hot / encode.

The **Preprocess tab is preview-only (dry-run)**: `POST /api/preprocess` returns a preview + step log and does not mutate stored datasets. Copy the same operations into the Train tab or a config file to apply them when training.

### Background jobs

`api/jobs.py` stores jobs in **SQLite** (`data/jobs.db`, gitignored; last 200 kept) and mirrors them in an in-process dict for fast reads. `POST /api/jobs/train` (and `POST /api/config/run`) creates a job, launches `asyncio.create_task` which runs `run_training` in the threadpool, and persists stage/result/error updates. Clients poll `GET /api/jobs/{id}`. On restart, jobs that were still `running` are marked `error` with detail “Interrupted by server restart”; finished job history remains queryable until pruned.

### MLflow

- Default tracking URI: `sqlite:///mlflow.db` (gitignored) so runs are queryable via `MlflowClient.search_runs`.
- `MLflowTracker` is a context manager: `__enter__` starts a run, `__exit__` ends it; `active_run_id` exposes the run id for API responses. Models are logged with `skops_trusted_types` covering XGBoost types.
- Training logs dataset provenance (`source`, `identifier`, `split`, `target`, split sizes, `random_state`) so `POST /api/evaluate` can reproduce the test split without the client re-specifying the dataset.
- `POST /api/evaluate` loads `runs:/{id}/model` via `mlflow.pyfunc.load_model`, rebuilds the split from logged params, and computes task metrics.
- `GET /api/runs/{id}/model` downloads the `model/` artifact subtree as a zip.
- Run management: rename, soft-delete, restore, full details (duration, artifacts, metric history). Endpoints accept optional `tracking_uri` query param.
- **Managed UI** (`api/mlflow_ui.py`): `POST /api/mlflow/ui/start` spawns `mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000` (port overridable, default `MLFLOW_UI_PORT`); stdout/stderr go to `mlflow-ui.log`. The full sqlite URI is passed as-is (a bare file path would make MLflow treat the store as a filesystem backend and fail). Stop/status are process-aware.
- **Storage** (`GET /api/storage`): reports project root, tracking URI/DB path, artifact root (from experiment `artifact_location`, else `mlruns/`), `data/uploads/`, `data/versions/`.
- **Predict** (`GET /api/predict/schema`, `POST /api/predict`): loads a logged run's model (sklearn loader first, pyfunc fallback) and the logged test-frame schema, returns feature metadata/classes, and predicts from a feature map. Classification responses include per-class probabilities.

### Error mapping

Centralized in `api/errors.py` so routes raise plain domain exceptions:

- `FileNotFoundError` → 404 (missing local file, missing run)
- `ModelError` and friends (`ValueError`, `KeyError`, `TypeError`, `AttributeError`) → 400
- `fastapi.HTTPException` → passed through with `detail` body

## Frontend

React SPA (JSX), built with Vite into `static/`. Dev server proxies `/api` to `:8000`.

```
frontend/src/
├── main.jsx              entry
├── App.jsx               shell: header, 7 tabs, datasets/models bootstrap
├── styles.css            theme, layout, tabs, tables, config/predict/storage panels
├── lib/
│   ├── api.js            fetch helpers (api/postJson)
│   └── preprocessOps.js  operation catalog
├── components/
│   ├── forms.jsx         Status, EmptyState, reusable form bits
│   └── PipelineGraph.jsx operations pipeline visualization
└── views/
    ├── Train.jsx         schema-driven param forms, config panel, background job polling, results, "Try this model"
    ├── Config.jsx        export ConfigPanel (collapsible YAML/JSON load/validate/apply)
    ├── Preprocess.jsx    operation builder + live preview + op log
    ├── Datasets.jsx      registry table, arbitrary preview, file upload
    ├── DataVersions.jsx  snapshot create/list/preview/diff/restore/delete
    ├── Runs.jsx          run history, detail, rename/delete/restore/evaluate/download, storage paths, model location
    ├── Predict.jsx       pick a logged run, fill feature schema, predict + probabilities
    └── Monitoring.jsx    MLflow status, start/stop UI, run count
```

- App bootstraps by fetching `/api/datasets` and `/api/models`.
- Training uses the job API: start → poll → render result (or error), with stage list shown live.
- Config lives inside the Train tab (expandable panel): validate, validate file, apply normalized request to the form, or validate & train.
- "Where are models?" on Runs opens the storage path panel; "Try this model" jumps to Predict with the run preselected.

## Data flow (end to end)

```
UI source+identifier ─► POST /api/preprocess ─► Preprocessor ops ─► preview + op log
UI / config YAML     ─► POST /api/jobs/train ─► job store ─► GET /api/jobs/{id}
                                          └─► MLflow (optional) ─► mlruns/ + mlflow.db
Run history ─► GET /api/runs ─► POST /api/evaluate ─► metrics on reproduced test split
                           ├─► GET /api/runs/{id}/model ─► zip download
                           └─► GET /api/predict/schema ─► POST /api/predict
Dataset    ─► POST /api/data-versions ─► data/versions/{name}/{id}/ + index.json
```

## Configuration & tooling

- Python 3.13, managed by **uv**; package installable (hatchling) with a `pytrain` console script (`api/cli.py` → uvicorn, `--host/--port/--reload`, or `PYTRAIN_HOST`/`PYTRAIN_PORT`).
- Direct deps: fastapi, uvicorn, pandas, scikit-learn, xgboost-cpu, mlflow, datasets, huggingface-hub, python-multipart, pyyaml. Optional extra `formats` adds openpyxl/xlrd/lxml/pyreadstat.
- **ruff** (lint + format), **pytest** (325 tests), **GitHub Actions** CI (`.github/workflows/ci.yml`).
- Frontend: React + Vite (`npm run build` → `static/`); never run the Vite build in parallel with pytest (emptyOutDir wipes `static/`).
- No auth, no multi-user state, no external services — designed as a local tool.

## Design decisions

1. **Async at the edges, sync in the core** — ML libraries are sync and thread-bound; `async def` + threadpool gives non-blocking I/O without rewriting the stack.
2. **Source/identifier over dataset names** — names are a convenience layer; anything resolvable by name is also addressable directly (including HF downloads and arbitrary local paths).
3. **SQLite jobs + sqlite MLflow** — zero external services; experiment history in `mlflow.db`, job history in `data/jobs.db`.
4. **Schema-driven UI** — `GET /api/models` returns parameter schemas; the train form and config validation are generated from them.
5. **Preprocessing precedence** — user `TrainRequest.operations` run first when provided; `prepare_xy` feature preparation always runs next. The Preprocess tab is preview-only (dry-run).
6. **Backend is source of truth** — no fake progress/metrics/registry data in the UI; stages and paths come from the API.
7. **sklearn + XGBoost only (v1)** — no-code tabular playground; deep learning / LLM fine-tuning deferred to a later release (in-app predict playground is the one serving-adjacent feature).

## Extension points

- SSE/WebSocket progress instead of job polling
- Auth + multi-user experiments
- Deep learning + LLM fine-tuning + Hugging Face integration (next major scope)
- Additional tabular trainers (LightGBM/CatBoost, early stopping for XGBoost)
- Batch CSV prediction endpoint
- Docker image + compose for deployment
