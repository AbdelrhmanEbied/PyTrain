from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.errors import register_error_handlers
from api.routes import (
    config_router,
    data_versions_router,
    datasets_router,
    models_router,
    monitoring_router,
    runs_router,
    training_router,
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="PyTrain", version="1.0.0")
    register_error_handlers(app)
    app.include_router(training_router)
    app.include_router(runs_router)
    app.include_router(datasets_router)
    app.include_router(models_router)
    app.include_router(monitoring_router)
    app.include_router(config_router)
    app.include_router(data_versions_router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
