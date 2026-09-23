from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

UI_PORT = int(os.environ.get("MLFLOW_UI_PORT", "5000"))
UI_URL = f"http://127.0.0.1:{UI_PORT}"
BACKEND_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

_process: subprocess.Popen[bytes] | None = None


def _ui_up(timeout: float = 1.0) -> bool:
    try:
        req = urllib.request.Request(UI_URL, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return 200 <= getattr(res, "status", 200) < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _ensure_backend_dir() -> None:
    if BACKEND_URI.startswith("sqlite:///"):
        path = BACKEND_URI.removeprefix("sqlite:///")
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)


def start_mlflow_ui(port: int | None = None) -> dict[str, Any]:
    global _process, UI_PORT, UI_URL
    if port is not None and port != UI_PORT:
        UI_PORT = port
        UI_URL = f"http://127.0.0.1:{UI_PORT}"

    if _ui_up():
        return {"started": False, "already_running": True, "ui_url": UI_URL, "pid": None}

    if _process is not None and _process.poll() is None:
        for _ in range(20):
            if _ui_up():
                return {
                    "started": False,
                    "already_running": True,
                    "ui_url": UI_URL,
                    "pid": _process.pid,
                }
            time.sleep(0.25)
        return {
            "started": False,
            "already_running": True,
            "ui_url": UI_URL,
            "pid": _process.pid,
            "pending": True,
        }

    _ensure_backend_dir()
    cmd = [
        "mlflow",
        "server",
        "--backend-store-uri",
        BACKEND_URI,
        "--host",
        "127.0.0.1",
        "--port",
        str(UI_PORT),
    ]
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    log_path = Path("mlflow-ui.log")
    log_file = open(log_path, "ab")
    try:
        _process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    finally:
        log_file.close()

    for _ in range(40):
        if _process.poll() is not None:
            break
        if _ui_up():
            return {
                "started": True,
                "already_running": False,
                "ui_url": UI_URL,
                "pid": _process.pid,
            }
        time.sleep(0.25)

    if _ui_up():
        return {
            "started": True,
            "already_running": False,
            "ui_url": UI_URL,
            "pid": _process.pid if _process else None,
        }

    detail = "MLflow UI did not become ready. Check mlflow-ui.log."
    if _process is not None and _process.poll() is not None:
        detail = f"MLflow UI exited with code {_process.returncode}. Check mlflow-ui.log."
    raise RuntimeError(detail)


def stop_mlflow_ui() -> dict[str, Any]:
    global _process
    if _process is None or _process.poll() is not None:
        _process = None
        return {"stopped": False, "was_running": False, "ui_url": UI_URL}
    try:
        os.killpg(os.getpgid(_process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        _process.terminate()
    try:
        _process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(_process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            _process.kill()
        _process.wait(timeout=3)
    was_running = True
    _process = None
    return {"stopped": True, "was_running": was_running, "ui_url": UI_URL}


def mlflow_ui_status() -> dict[str, Any]:
    pid = _process.pid if _process is not None and _process.poll() is None else None
    return {
        "ui_url": UI_URL,
        "server_up": _ui_up(),
        "managed_pid": pid,
        "managed": pid is not None,
        "port": UI_PORT,
    }
