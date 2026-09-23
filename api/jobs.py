from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from starlette.concurrency import run_in_threadpool

DB_PATH = Path("data/jobs.db")
MAX_JOBS = 200


@dataclass
class Job:
    job_id: str
    status: str = "running"
    stage: str = "queued"
    detail: str | None = None
    pipeline: list[str] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)
    result: Any = None
    error: str | None = None

    def set_stage(self, stage: str, detail: str | None = None) -> None:
        self.stage = stage
        self.detail = detail
        self.stages.append({"stage": stage, "detail": detail, "at": time.time()})


JOBS: dict[str, Job] = {}
_loaded = False


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                detail TEXT,
                pipeline TEXT NOT NULL,
                stages TEXT NOT NULL,
                result TEXT,
                error TEXT,
                updated_at REAL NOT NULL
            )
            """
        )


def _persist(job: Job) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs (job_id, status, stage, detail, pipeline, stages, result, error, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                status=excluded.status,
                stage=excluded.stage,
                detail=excluded.detail,
                pipeline=excluded.pipeline,
                stages=excluded.stages,
                result=excluded.result,
                error=excluded.error,
                updated_at=excluded.updated_at
            """,
            (
                job.job_id,
                job.status,
                job.stage,
                job.detail,
                json.dumps(job.pipeline),
                json.dumps(job.stages),
                json.dumps(job.result) if job.result is not None else None,
                job.error,
                time.time(),
            ),
        )
        conn.execute(
            """
            DELETE FROM jobs WHERE job_id NOT IN (
                SELECT job_id FROM jobs ORDER BY updated_at DESC LIMIT ?
            )
            """,
            (MAX_JOBS,),
        )


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        job_id=row["job_id"],
        status=row["status"],
        stage=row["stage"],
        detail=row["detail"],
        pipeline=json.loads(row["pipeline"] or "[]"),
        stages=json.loads(row["stages"] or "[]"),
        result=json.loads(row["result"]) if row["result"] else None,
        error=row["error"],
    )


def _load_all() -> None:
    global _loaded
    if _loaded:
        return
    _init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY updated_at DESC").fetchall()
    for row in rows:
        job = _row_to_job(row)
        if job.status == "running":
            job.status = "error"
            job.error = "Interrupted by server restart"
            job.set_stage("error", "Interrupted by server restart")
            _persist(job)
        JOBS[job.job_id] = job
    _loaded = True


def get_job(job_id: str) -> Job | None:
    _load_all()
    return JOBS.get(job_id)


async def submit(
    fn: Callable[..., Any],
    *args: Any,
    stages: list[str] | None = None,
    **kwargs: Any,
) -> str:
    _load_all()
    job_id = uuid4().hex
    pipeline = list(stages) if stages else []
    job = Job(job_id=job_id, pipeline=pipeline)
    job.set_stage(pipeline[0] if pipeline else "running")
    JOBS[job_id] = job
    _persist(job)

    async def _run() -> None:
        try:

            def call() -> Any:
                if stages is not None:

                    def on_stage(stage: str, detail: str | None = None) -> None:
                        job.set_stage(stage, detail)
                        _persist(job)

                    return fn(*args, on_stage=on_stage, **kwargs)
                return fn(*args, **kwargs)

            result = await run_in_threadpool(call)
            job.result = result
            job.status = "done"
            job.set_stage("done")
            _persist(job)
        except Exception as exc:
            job.error = str(exc)
            job.status = "error"
            job.set_stage("error", str(exc))
            _persist(job)

    asyncio.create_task(_run())
    return job_id
