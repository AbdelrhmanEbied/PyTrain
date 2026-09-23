from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DatasetSource = Literal["huggingface", "local", "sklearn"]


class DatasetRef(BaseModel):
    source: DatasetSource | None = None
    identifier: str | None = None
    split: str | None = None
    dataset: str | None = None


class DatasetInfo(BaseModel):
    name: str
    source: DatasetSource
    identifier: str
    split: str | None = None
    target: str | None = None


class DatasetPreview(BaseModel):
    dataset: str
    rows: int
    columns: list[str]
    dtypes: dict[str, str]
    missing: dict[str, int]
    preview: list[dict[str, Any]]
    duplicates: int = 0
    columns_stats: list[dict[str, Any]] = Field(default_factory=list)
    class_distribution: dict[str, float] | None = None
    target: str | None = None


class ParamConfigIn(BaseModel):
    grid: dict[str, list[Any]]
    cv: int = 5
    scoring: str | None = None
    refit: bool = True
    n_jobs: int | None = None
    verbose: int = 0


class TrainRequest(DatasetRef):
    target: str | None = None
    model_name: str = "logistic_regression"
    task: Literal["classification", "regression"] = "classification"
    params: dict[str, Any] = Field(default_factory=dict)
    param_config: ParamConfigIn | None = None
    test_size: float = 0.2
    validation_size: float | None = 0.1
    stratify: bool = True
    random_state: int = 42
    log_mlflow: bool = False
    experiment_name: str = "pytrain"
    tracking_uri: str | None = None
    operations: list[dict[str, Any]] = Field(default_factory=list)


class TrainResponse(BaseModel):
    model_info: dict[str, Any]
    params: dict[str, Any]
    best_params: dict[str, Any] | None
    cv_results: dict[str, Any] | None
    train_metrics: dict[str, float]
    val_metrics: dict[str, float] | None
    test_metrics: dict[str, float]
    train_time: float
    eval_time: float
    n_train: int
    n_val: int
    n_test: int
    n_features: int
    logged_to_mlflow: bool
    mlflow_run_id: str | None
    pipeline: list[dict[str, Any]] = Field(default_factory=list)
    preprocess_steps: list[dict[str, Any]] = Field(default_factory=list)
    preprocess_ops: list[dict[str, Any]] = Field(default_factory=list)


class JobStage(BaseModel):
    stage: str
    detail: str | None = None
    at: float


class JobStatus(BaseModel):
    job_id: str
    status: Literal["running", "done", "error"]
    stage: str = "queued"
    detail: str | None = None
    pipeline: list[str] = Field(default_factory=list)
    stages: list[JobStage] = Field(default_factory=list)
    result: TrainResponse | None = None
    error: str | None = None


class PreprocessRequest(DatasetRef):
    operations: list[dict[str, Any]] = Field(default_factory=list)
    n_rows: int = 20
    target: str | None = None


class PreprocessStep(BaseModel):
    operation: str
    args: dict[str, Any] = Field(default_factory=dict)
    rows: int
    cols: int


class PreprocessResponse(BaseModel):
    dataset: str
    rows: int
    columns: list[str]
    dtypes: dict[str, str]
    missing: dict[str, int]
    preview: list[dict[str, Any]]
    operations: list[dict[str, Any]]
    steps: list[PreprocessStep]
    before_shape: tuple[int, int]
    after_shape: tuple[int, int]
    duplicates: int = 0
    columns_stats: list[dict[str, Any]] = Field(default_factory=list)
    class_distribution: dict[str, float] | None = None
    target: str | None = None


class UploadResponse(BaseModel):
    name: str
    source: Literal["local"]
    identifier: str
    rows: int
    columns: list[str]


class EvaluateRequest(DatasetRef):
    run_id: str
    target: str | None = None
    test_size: float = 0.2
    validation_size: float | None = 0.1
    stratify: bool = True
    random_state: int = 42
    tracking_uri: str | None = None


class EvaluateResponse(BaseModel):
    run_id: str
    metrics: dict[str, float]
    n_test: int
    eval_time: float
    task: str = "classification"
    confusion_matrix: dict[str, Any] | None = None
    samples: list[dict[str, float]] = Field(default_factory=list)


class RunInfo(BaseModel):
    run_id: str
    name: str | None = None
    experiment_id: str | None = None
    status: str | None = None
    start_time: int | None = None
    end_time: int | None = None
    params: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    pipeline: list[dict[str, Any]] = Field(default_factory=list)


class MlflowStatus(BaseModel):
    ui_url: str
    tracking_uri: str
    server_up: bool
    experiment_name: str = "pytrain"
    run_count: int = 0


class MlflowUiControl(BaseModel):
    started: bool = False
    already_running: bool = False
    stopped: bool = False
    was_running: bool = False
    ui_url: str
    pid: int | None = None
    server_up: bool = False
    managed: bool = False
    port: int | None = None


class ConfigValidateRequest(BaseModel):
    content: str
    format: Literal["json", "yaml", "yml"] | None = None


class ConfigValidateResponse(BaseModel):
    valid: bool
    request: dict[str, Any]
    normalized_yaml: str
    normalized_json: str


class RunRenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class RunDetail(RunInfo):
    duration_ms: int | None = None
    lifecycle_stage: str | None = None
    artifact_uri: str | None = None
    user_id: str | None = None
    preprocess_ops: list[dict[str, Any]] = Field(default_factory=list)
    preprocess_steps: list[dict[str, Any]] = Field(default_factory=list)
    split_config: dict[str, Any] = Field(default_factory=dict)
    model_hyperparams: dict[str, Any] = Field(default_factory=dict)
    grid_config: dict[str, Any] | None = None
    dataset_info: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metric_history: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class DataVersionCreate(BaseModel):
    dataset: str | None = None
    source: str | None = None
    identifier: str | None = None
    split: str | None = None
    target: str | None = None
    message: str = ""
    registered_name: str | None = None


class DataVersionInfo(BaseModel):
    id: str
    dataset: str
    version: int
    created_at: float
    message: str = ""
    source: str | None = None
    identifier: str | None = None
    split: str | None = None
    target: str | None = None
    format: str | None = None
    path: str | None = None
    dir: str | None = None
    rows: int = 0
    columns: int = 0
    column_names: list[str] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)
    missing: dict[str, int] = Field(default_factory=dict)
    missing_total: int = 0
    duplicates: int = 0
    hash: str = ""


class DataVersionDiff(BaseModel):
    a: dict[str, Any]
    b: dict[str, Any]
    same_hash: bool
    row_delta: int
    col_delta: int
    added_columns: list[str]
    removed_columns: list[str]
    changed_columns: list[str]
    n_changed_columns: int


class DataVersionRestore(BaseModel):
    name: str
    source: Literal["local"]
    identifier: str
    rows: int
    columns: list[str]
    from_version: str
    version: int | None = None
    target: str | None = None


class StorageInfo(BaseModel):
    project_root: str
    tracking_uri: str
    tracking_db: str | None = None
    artifact_root: str | None = None
    data_uploads: str
    data_versions: str
    ui_url: str | None = None


class PredictSchemaResponse(BaseModel):
    run_id: str
    task: str
    target: str | None = None
    model_name: str | None = None
    features: list[dict[str, Any]]
    n_features: int
    classes: list[Any] | None = None
    artifact_uri: str | None = None


class PredictRequest(BaseModel):
    run_id: str
    features: dict[str, Any]
    tracking_uri: str | None = None


class PredictResponse(BaseModel):
    run_id: str
    task: str
    prediction: Any
    probabilities: dict[str, float] | None = None
    classes: list[Any] | None = None
