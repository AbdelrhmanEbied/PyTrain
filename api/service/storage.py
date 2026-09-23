from __future__ import annotations

import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from api.service.runs import DEFAULT_TRACKING_URI, _mlflow_client


def storage_info(tracking_uri: str | None = None) -> dict[str, Any]:
    project_root = Path.cwd().resolve()
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    tracking_db: str | None = None
    if uri.startswith("sqlite:///"):
        raw = uri.removeprefix("sqlite:///")
        if raw and raw != ":memory:":
            path = Path(raw)
            tracking_db = str(
                path.resolve() if path.is_absolute() else (project_root / path).resolve()
            )

    artifact_root: str | None = None
    try:
        client = _mlflow_client(uri)
        experiments = client.search_experiments()
        if experiments:
            loc = experiments[0].artifact_location or ""
            if loc.startswith("file://"):
                artifact_root = urllib.request.url2pathname(urllib.parse.urlparse(loc).path)
            elif loc:
                artifact_root = loc
    except Exception:
        artifact_root = None
    if not artifact_root:
        artifact_root = str(project_root / "mlruns")

    ui = (
        os.environ.get("MLFLOW_UI_URL")
        or os.environ.get("MLFLOW_TRACKING_UI")
        or "http://127.0.0.1:5000"
    ).rstrip("/")

    return {
        "project_root": str(project_root),
        "tracking_uri": uri,
        "tracking_db": tracking_db,
        "artifact_root": artifact_root,
        "data_uploads": str((project_root / "data" / "uploads").resolve()),
        "data_versions": str((project_root / "data" / "versions").resolve()),
        "ui_url": ui,
    }
