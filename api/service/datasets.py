from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from data.dataloader import READERS, SKLEARN_DATASETS, DataLoader, Dataset

TITANIC_PATH = Path("data/test_datasets/titanic.csv")
TITANIC_DROP = ["boat", "body", "home.dest", "cabin", "ticket", "name"]
UPLOADS_DIR = Path("data/uploads")

DEFAULT_TARGETS = {
    "titanic": "survived",
    "iris": "target",
    "wine": "target",
    "breast_cancer": "target",
    "diabetes": "target",
    "digits": "target",
    "linnerud": "target",
}


def list_datasets() -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = [
        {
            "name": "titanic",
            "source": "local",
            "identifier": str(TITANIC_PATH),
            "split": None,
            "target": "survived",
        }
    ]
    for name in sorted(SKLEARN_DATASETS):
        datasets.append(
            {
                "name": name,
                "source": "sklearn",
                "identifier": name,
                "split": None,
                "target": "target",
            }
        )
    for path in sorted(UPLOADS_DIR.glob("*")) if UPLOADS_DIR.exists() else []:
        if path.is_file() and path.suffix.lower() in READERS:
            datasets.append(
                {
                    "name": path.stem,
                    "source": "local",
                    "identifier": str(path),
                    "split": None,
                    "target": DEFAULT_TARGETS.get(path.stem),
                }
            )
    return datasets


def find_dataset(name: str) -> dict[str, Any] | None:
    for entry in list_datasets():
        if entry["name"] == name:
            return entry
    return None


def resolve_ref(
    source: str | None,
    identifier: str | None,
    split: str | None,
    dataset: str | None,
) -> Dataset:
    if source is not None and identifier is not None:
        return Dataset(source=source, identifier=identifier, split=split)
    name = dataset or "titanic"
    entry = find_dataset(name)
    if entry is None:
        if Path(name).exists():
            return Dataset(source="local", identifier=name, split=split)
        available = [d["name"] for d in list_datasets()]
        raise ValueError(f"Unknown dataset: {name!r}. Available: {available}")
    return Dataset(
        source=entry["source"],
        identifier=entry["identifier"],
        split=entry.get("split"),
    )


def dataset_label(ref: Dataset) -> str:
    if ref.source == "local":
        return Path(str(ref.identifier)).stem
    return str(ref.identifier)


def load_frame(ref: Dataset) -> pd.DataFrame:
    loader = DataLoader(ref)
    return loader.load()


def load_dataframe(dataset: str) -> pd.DataFrame:
    entry = find_dataset(dataset)
    if entry is not None:
        return load_frame(
            Dataset(
                source=entry["source"],
                identifier=entry["identifier"],
                split=entry.get("split"),
            )
        )
    if Path(dataset).exists():
        return load_frame(Dataset(source="local", identifier=dataset))
    available = [d["name"] for d in list_datasets()]
    raise ValueError(f"Unknown dataset: {dataset!r}. Available: {available}")


def json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


def _json_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return value


def column_stats(df: pd.DataFrame) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    n_rows = max(len(df), 1)
    for col in df.columns:
        series = df[col]
        entry: dict[str, Any] = {
            "column": str(col),
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean()) * 100, 2),
            "unique": int(series.nunique(dropna=True)),
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "top": None,
            "top_pct": None,
        }
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            valid = series.dropna()
            if len(valid):
                entry["mean"] = float(valid.mean())
                entry["std"] = float(valid.std()) if len(valid) > 1 else 0.0
                entry["min"] = float(valid.min())
                entry["max"] = float(valid.max())
        else:
            counts = series.value_counts(dropna=True)
            if len(counts):
                entry["top"] = _json_scalar(counts.index[0])
                entry["top_pct"] = round(float(counts.iloc[0]) / n_rows * 100, 2)
        stats.append(entry)
    return stats


def class_distribution(df: pd.DataFrame, target: str) -> dict[str, float] | None:
    if target not in df.columns:
        return None
    series = df[target]
    if len(series) == 0:
        return None
    counts = series.value_counts(dropna=False).head(32)
    return {str(_json_scalar(k)): round(float(v) / len(series) * 100, 2) for k, v in counts.items()}


def preview_frame(
    df: pd.DataFrame,
    label: str,
    n_rows: int = 20,
    target: str | None = None,
) -> dict[str, Any]:
    resolved_target = target if target and target in df.columns else None
    if resolved_target is None and target is None:
        for candidate in DEFAULT_TARGETS.values():
            if candidate in df.columns:
                resolved_target = candidate
                break
        if resolved_target is None:
            for candidate in ("target", "label", "class", "y"):
                if candidate in df.columns:
                    resolved_target = candidate
                    break
    return {
        "dataset": label,
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "preview": json_records(df.head(n_rows)),
        "duplicates": int(df.duplicated().sum()),
        "columns_stats": column_stats(df),
        "class_distribution": class_distribution(df, resolved_target) if resolved_target else None,
        "target": resolved_target,
    }


def load_dataset_preview(
    source: str | None = None,
    identifier: str | None = None,
    split: str | None = None,
    dataset: str | None = None,
    n_rows: int = 20,
    target: str | None = None,
) -> dict[str, Any]:
    ref = resolve_ref(source, identifier, split, dataset)
    df = load_frame(ref)
    if dataset is not None and source is None:
        label = dataset
    else:
        label = dataset_label(ref)
    if target is None and dataset is not None:
        entry = find_dataset(dataset)
        if entry is not None:
            target = entry.get("target")
    return preview_frame(df, label, n_rows=n_rows, target=target)


def default_target(dataset: str) -> str:
    return DEFAULT_TARGETS.get(dataset, "target")


def save_upload(filename: str, content: bytes) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix not in READERS:
        raise ValueError(
            f"Unsupported file format: {suffix or '(none)'}. Supported: {sorted(READERS)}"
        )
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)
    path = UPLOADS_DIR / safe
    path.write_bytes(content)
    df = load_frame(Dataset(source="local", identifier=path))
    return {
        "name": path.stem,
        "source": "local",
        "identifier": str(path),
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
    }
