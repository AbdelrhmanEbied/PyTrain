from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from api.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    PredictRequest,
    PredictResponse,
    PredictSchemaResponse,
    RunDetail,
    RunInfo,
    RunRenameRequest,
    StorageInfo,
)
from api.service import (
    delete_run,
    evaluate_run,
    get_run,
    get_run_details,
    list_runs,
    predict_run,
    predict_schema,
    rename_run,
    restore_run,
    storage_info,
)

router = APIRouter(prefix="/api", tags=["runs"])


@router.get("/runs", response_model=list[RunInfo])
async def get_runs(
    limit: int = 50,
    tracking_uri: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    return await run_in_threadpool(list_runs, tracking_uri, limit)


@router.get("/runs/{run_id}", response_model=RunInfo)
async def get_run_detail(
    run_id: str,
    tracking_uri: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return await run_in_threadpool(get_run, run_id, tracking_uri)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc


@router.get("/runs/{run_id}/details", response_model=RunDetail)
async def get_run_full(
    run_id: str,
    tracking_uri: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return await run_in_threadpool(get_run_details, run_id, tracking_uri)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc


@router.patch("/runs/{run_id}", response_model=RunDetail)
async def patch_run(
    run_id: str,
    req: RunRenameRequest,
    tracking_uri: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return await run_in_threadpool(rename_run, run_id, req.name, tracking_uri)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc


@router.delete("/runs/{run_id}")
async def remove_run(
    run_id: str,
    tracking_uri: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return await run_in_threadpool(delete_run, run_id, tracking_uri)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc


@router.post("/runs/{run_id}/restore")
async def undo_delete_run(
    run_id: str,
    tracking_uri: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return await run_in_threadpool(restore_run, run_id, tracking_uri)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc


@router.get("/runs/{run_id}/model")
async def download_model(
    run_id: str,
    tracking_uri: str | None = Query(default=None),
) -> FileResponse:
    def _fetch() -> str:
        import mlflow

        from api.service import DEFAULT_TRACKING_URI

        mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
        mlflow.get_run(run_id)
        base = Path(tempfile.mkdtemp())
        try:
            mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="model", dst_path=str(base)
            )
        except Exception as exc:
            raise FileNotFoundError(f"No model artifact for run {run_id}: {exc}") from exc
        model_dir = base / "model"
        if not (model_dir / "MLmodel").exists():
            raise FileNotFoundError(f"No model artifact for run {run_id}")
        return shutil.make_archive(str(base / f"{run_id}-model"), "zip", root_dir=model_dir)

    try:
        archive = await run_in_threadpool(_fetch)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(archive, media_type="application/zip", filename=f"{run_id}-model.zip")


@router.post("/evaluate", response_model=EvaluateResponse)
async def post_evaluate(req: EvaluateRequest) -> dict[str, Any]:
    return await run_in_threadpool(evaluate_run, req)


@router.get("/storage", response_model=StorageInfo)
async def get_storage(
    tracking_uri: str | None = Query(default=None),
) -> dict[str, Any]:
    return await run_in_threadpool(storage_info, tracking_uri)


@router.get("/predict/schema", response_model=PredictSchemaResponse)
async def get_predict_schema(
    run_id: str = Query(...),
    tracking_uri: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return await run_in_threadpool(predict_schema, run_id, tracking_uri)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        if "not found" in str(exc).lower():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc
        raise


@router.post("/predict", response_model=PredictResponse)
async def post_predict(req: PredictRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(predict_run, req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        if "not found" in str(exc).lower():
            raise HTTPException(status_code=404, detail=f"Run not found: {req.run_id}") from exc
        raise
