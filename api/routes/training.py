from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from api import jobs
from api.schemas import JobStatus, TrainRequest, TrainResponse
from api.service import run_training

router = APIRouter(prefix="/api", tags=["training"])


@router.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest) -> dict[str, Any]:
    return await run_in_threadpool(run_training, req)


@router.post("/jobs/train")
async def start_training_job(req: TrainRequest) -> dict[str, str]:
    stages = ["queued", "loading"]
    if req.operations:
        stages.append("preprocessing")
    stages.extend(["preparing", "training", "evaluating"])
    if req.log_mlflow:
        stages.append("logging")
    stages.append("done")
    job_id = await jobs.submit(run_training, req, stages=stages)
    return {"job_id": job_id, "status": "running"}


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def training_job(job_id: str) -> dict[str, Any]:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "detail": job.detail,
        "pipeline": job.pipeline,
        "stages": job.stages,
        "result": job.result,
        "error": job.error,
    }
