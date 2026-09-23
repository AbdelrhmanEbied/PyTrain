from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from starlette.concurrency import run_in_threadpool

from api import mlflow_ui
from api.schemas import MlflowUiControl
from api.service import mlflow_status

router = APIRouter(prefix="/api", tags=["monitoring"])


@router.get("/mlflow/status")
async def get_mlflow_status(
    tracking_uri: str | None = Query(default=None),
    experiment_name: str = Query(default="pytrain"),
    ui_url: str | None = Query(default=None),
) -> dict[str, Any]:
    return await run_in_threadpool(mlflow_status, tracking_uri, experiment_name, ui_url)


@router.get("/mlflow/ui/status", response_model=MlflowUiControl)
async def get_ui_status() -> dict[str, Any]:
    return await run_in_threadpool(mlflow_ui.mlflow_ui_status)


@router.post("/mlflow/ui/start", response_model=MlflowUiControl)
async def post_ui_start(
    port: int | None = Query(default=None, ge=1024, le=65535),
) -> dict[str, Any]:
    return await run_in_threadpool(mlflow_ui.start_mlflow_ui, port)


@router.post("/mlflow/ui/stop", response_model=MlflowUiControl)
async def post_ui_stop() -> dict[str, Any]:
    return await run_in_threadpool(mlflow_ui.stop_mlflow_ui)
