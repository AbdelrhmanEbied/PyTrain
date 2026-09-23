from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from api.pipeline_config import config_to_train_request, parse_config_text, validate_config_text
from api.schemas import ConfigValidateRequest, ConfigValidateResponse, TrainRequest, TrainResponse
from api.service import run_training

router = APIRouter(prefix="/api", tags=["config"])


@router.post("/config/validate", response_model=ConfigValidateResponse)
async def validate_config(req: ConfigValidateRequest) -> dict[str, Any]:
    return await run_in_threadpool(validate_config_text, req.content, req.format)


@router.post("/config/validate-file", response_model=ConfigValidateResponse)
async def validate_config_file(
    file: UploadFile = File(...),
    format: str | None = Form(default=None),
) -> dict[str, Any]:
    raw = (await file.read()).decode("utf-8", errors="replace")
    fmt = format
    if not fmt and file.filename:
        name = file.filename.lower()
        if name.endswith(".json"):
            fmt = "json"
        elif name.endswith((".yaml", ".yml")):
            fmt = "yaml"
    return await run_in_threadpool(validate_config_text, raw, fmt)


@router.post("/config/to-request")
async def config_to_request(req: ConfigValidateRequest) -> dict[str, Any]:
    def _convert() -> dict[str, Any]:
        data = parse_config_text(req.content, req.format)
        request = config_to_train_request(data)
        return request.model_dump()

    return await run_in_threadpool(_convert)


@router.post("/config/run")
async def run_from_config(req: ConfigValidateRequest) -> dict[str, Any]:
    def _prepare() -> TrainRequest:
        data = parse_config_text(req.content, req.format)
        return config_to_train_request(data)

    train_req = await run_in_threadpool(_prepare)
    stages = ["queued", "loading"]
    if train_req.operations:
        stages.append("preprocessing")
    stages.extend(["preparing", "training", "evaluating"])
    if train_req.log_mlflow:
        stages.append("logging")
    stages.append("done")
    from api import jobs

    job_id = await jobs.submit(run_training, train_req, stages=stages)
    return {"job_id": job_id, "status": "running", "request": train_req.model_dump()}


@router.post("/config/run-sync", response_model=TrainResponse)
async def run_from_config_sync(req: ConfigValidateRequest) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        data = parse_config_text(req.content, req.format)
        train_req = config_to_train_request(data)
        return run_training(train_req)

    return await run_in_threadpool(_run)
