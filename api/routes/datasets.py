from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from api.schemas import (
    DatasetInfo,
    DatasetPreview,
    PreprocessRequest,
    PreprocessResponse,
    UploadResponse,
)
from api.service import list_datasets, load_dataset_preview, preprocess_dataset, save_upload

router = APIRouter(prefix="/api", tags=["datasets"])


@router.get("/datasets", response_model=list[DatasetInfo])
async def get_datasets() -> list[dict[str, Any]]:
    return list_datasets()


@router.post("/datasets/preview", response_model=DatasetPreview)
async def preview_by_ref(req: PreprocessRequest) -> dict[str, Any]:
    return await run_in_threadpool(
        load_dataset_preview,
        source=req.source,
        identifier=req.identifier,
        split=req.split,
        dataset=req.dataset,
        n_rows=req.n_rows,
        target=req.target,
    )


@router.post("/datasets/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    return await run_in_threadpool(save_upload, file.filename or "upload.csv", content)


@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(req: PreprocessRequest) -> dict[str, Any]:
    return await run_in_threadpool(preprocess_dataset, req)


@router.get("/datasets/{name}", response_model=DatasetPreview)
async def preview_named(name: str, n_rows: int = Query(default=20, ge=1, le=500)) -> dict[str, Any]:
    return await run_in_threadpool(load_dataset_preview, dataset=name, n_rows=n_rows)
