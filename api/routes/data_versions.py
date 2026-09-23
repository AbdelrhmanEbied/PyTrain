from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from starlette.concurrency import run_in_threadpool

from api import data_versions
from api.schemas import (
    DatasetPreview,
    DataVersionCreate,
    DataVersionDiff,
    DataVersionInfo,
    DataVersionRestore,
)

router = APIRouter(prefix="/api/data-versions", tags=["data-versions"])


@router.get("", response_model=list[DataVersionInfo])
async def list_versions(dataset: str | None = Query(default=None)) -> list[dict[str, Any]]:
    return await run_in_threadpool(data_versions.list_data_versions, dataset)


@router.post("", response_model=DataVersionInfo)
async def create_version(req: DataVersionCreate) -> dict[str, Any]:
    return await run_in_threadpool(
        data_versions.create_data_version,
        dataset=req.dataset,
        source=req.source,
        identifier=req.identifier,
        split=req.split,
        target=req.target,
        message=req.message,
        registered_name=req.registered_name,
    )


@router.get("/diff", response_model=DataVersionDiff)
async def diff_versions(a: str = Query(...), b: str = Query(...)) -> dict[str, Any]:
    return await run_in_threadpool(data_versions.diff_data_versions, a, b)


@router.get("/{version_id}", response_model=DataVersionInfo)
async def get_version(version_id: str) -> dict[str, Any]:
    return await run_in_threadpool(data_versions.get_data_version, version_id)


@router.get("/{version_id}/preview", response_model=DatasetPreview)
async def preview_version(
    version_id: str,
    n_rows: int = Query(default=20, ge=1, le=500),
) -> dict[str, Any]:
    return await run_in_threadpool(data_versions.preview_data_version, version_id, n_rows)


@router.post("/{version_id}/restore", response_model=DataVersionRestore)
async def restore_version(
    version_id: str, name: str | None = Query(default=None)
) -> dict[str, Any]:
    return await run_in_threadpool(data_versions.restore_data_version, version_id, name)


@router.delete("/{version_id}")
async def delete_version(version_id: str) -> dict[str, Any]:
    return await run_in_threadpool(data_versions.delete_data_version, version_id)
