from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pytrain",
        description="Run the PyTrain server (FastAPI + React UI).",
    )
    parser.add_argument("--host", default=os.environ.get("PYTRAIN_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PYTRAIN_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)
