from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from api.service import DEFAULT_TARGETS, find_dataset, load_frame, resolve_ref
from data.dataloader import Dataset

VERSIONS_DIR = Path("data/versions")


def _index_path() -> Path:
    return VERSIONS_DIR / "index.json"


def _load_index() -> list[dict[str, Any]]:
    path = _index_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _save_index(entries: list[dict[str, Any]]) -> None:
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    _index_path().write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _frame_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode())
    h.update(str(df.dtypes.astype(str).values.tolist()).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()


def _df_summary(df: pd.DataFrame) -> dict[str, Any]:
    missing = {str(k): int(v) for k, v in df.isna().sum().items()}
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "missing": missing,
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "hash": _frame_hash(df),
    }


def _version_dir(dataset: str, version_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in dataset)
    return VERSIONS_DIR / safe / version_id


def create_data_version(
    dataset: str | None = None,
    source: str | None = None,
    identifier: str | None = None,
    split: str | None = None,
    target: str | None = None,
    message: str = "",
    registered_name: str | None = None,
) -> dict[str, Any]:
    ref = resolve_ref(source, identifier, split, dataset)
    name = (
        registered_name
        or dataset
        or (Path(str(ref.identifier)).stem if ref.source == "local" else str(ref.identifier))
    )
    df = load_frame(ref)
    summary = _df_summary(df)
    version_id = uuid4().hex[:12]
    version = int(sum(1 for e in _load_index() if e.get("dataset") == name) + 1)
    directory = _version_dir(name, version_id)
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / "data.parquet"
    try:
        df.to_parquet(data_path, index=False)
        format_used = "parquet"
    except Exception:
        data_path = directory / "data.csv"
        df.to_csv(data_path, index=False)
        format_used = "csv"

    entry = {
        "id": version_id,
        "dataset": name,
        "version": version,
        "created_at": time.time(),
        "message": message or f"Snapshot of {name}",
        "source": ref.source,
        "identifier": str(ref.identifier),
        "split": ref.split,
        "target": target or DEFAULT_TARGETS.get(name),
        "format": format_used,
        "path": str(data_path),
        "dir": str(directory),
        **summary,
    }
    index = _load_index()
    index.append(entry)
    _save_index(index)
    return entry


def list_data_versions(dataset: str | None = None) -> list[dict[str, Any]]:
    entries = _load_index()
    if dataset:
        entries = [e for e in entries if e.get("dataset") == dataset]
    return sorted(entries, key=lambda e: e.get("created_at") or 0, reverse=True)


def get_data_version(version_id: str) -> dict[str, Any]:
    for entry in _load_index():
        if entry.get("id") == version_id:
            return entry
    raise FileNotFoundError(f"Data version not found: {version_id}")


def _load_version_frame(entry: dict[str, Any]) -> pd.DataFrame:
    path = Path(entry["path"])
    if not path.exists():
        raise FileNotFoundError(f"Version file missing: {path}")
    if entry.get("format") == "csv" or path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def delete_data_version(version_id: str) -> dict[str, Any]:
    entry = get_data_version(version_id)
    directory = Path(entry.get("dir") or Path(entry["path"]).parent)
    if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)
    index = [e for e in _load_index() if e.get("id") != version_id]
    _save_index(index)
    return {"deleted": True, "id": version_id, "dataset": entry.get("dataset")}


def diff_data_versions(a_id: str, b_id: str) -> dict[str, Any]:
    a = get_data_version(a_id)
    b = get_data_version(b_id)
    df_a = _load_version_frame(a)
    df_b = _load_version_frame(b)
    cols_a = list(df_a.columns)
    cols_b = list(df_b.columns)
    added_cols = [c for c in cols_b if c not in cols_a]
    removed_cols = [c for c in cols_a if c not in cols_b]
    common = [c for c in cols_a if c in cols_b]
    changed_cols = [c for c in common if not df_a[c].equals(df_b[c])]
    return {
        "a": {
            k: a.get(k) for k in ("id", "dataset", "version", "rows", "columns", "hash", "message")
        },
        "b": {
            k: b.get(k) for k in ("id", "dataset", "version", "rows", "columns", "hash", "message")
        },
        "same_hash": a.get("hash") == b.get("hash"),
        "row_delta": int(df_b.shape[0] - df_a.shape[0]),
        "col_delta": int(df_b.shape[1] - df_a.shape[1]),
        "added_columns": added_cols,
        "removed_columns": removed_cols,
        "changed_columns": changed_cols[:50],
        "n_changed_columns": len(changed_cols),
    }


def restore_data_version(version_id: str, as_name: str | None = None) -> dict[str, Any]:
    entry = get_data_version(version_id)
    df = _load_version_frame(entry)
    UPLOADS = Path("data/uploads")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    name = as_name or entry.get("dataset") or "restored"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    path = UPLOADS / f"{safe}.csv"
    df.to_csv(path, index=False)
    restored = {
        "name": safe,
        "source": "local",
        "identifier": str(path),
        "rows": int(df.shape[0]),
        "columns": [str(c) for c in df.columns],
        "from_version": version_id,
        "version": entry.get("version"),
        "target": entry.get("target") or DEFAULT_TARGETS.get(safe),
    }
    return restored


def preview_data_version(version_id: str, n_rows: int = 20) -> dict[str, Any]:
    from api.service import preview_frame

    entry = get_data_version(version_id)
    df = _load_version_frame(entry)
    label = f"{entry.get('dataset')}@v{entry.get('version')}"
    payload = preview_frame(df, label, n_rows=n_rows, target=entry.get("target"))
    payload["version"] = entry.get("version")
    payload["version_id"] = version_id
    payload["message"] = entry.get("message")
    payload["created_at"] = entry.get("created_at")
    payload["hash"] = entry.get("hash")
    return payload


def resolve_version_ref(version_id: str) -> Dataset:
    entry = get_data_version(version_id)
    path = Path(entry["path"])
    return Dataset(source="local", identifier=str(path), split=None)


def version_ref_dataset_entry(version_id: str) -> dict[str, Any]:
    entry = get_data_version(version_id)
    return {
        "name": f"{entry.get('dataset')}_v{entry.get('version')}",
        "source": "local",
        "identifier": entry["path"],
        "split": None,
        "target": entry.get("target"),
        "version_id": version_id,
    }


def ensure_dataset_exists(dataset: str) -> dict[str, Any]:
    entry = find_dataset(dataset)
    if entry is None:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    return entry
