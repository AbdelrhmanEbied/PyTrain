from __future__ import annotations

from typing import Any

import pandas as pd

from api.schemas import PreprocessRequest
from api.service.datasets import (
    TITANIC_DROP,
    dataset_label,
    load_frame,
    preview_frame,
    resolve_ref,
)
from data.preprocessing import Preprocessor, PreprocessorConfig
from models.base import Task

PREPROCESSOR_METHODS = {
    "drop_columns",
    "rename_columns",
    "drop_duplicates",
    "drop_missing_rows",
    "fill_missing",
    "convert_dtype",
    "parse_dates",
    "clean_numeric",
    "filter_rows",
    "shuffle",
    "sample_rows",
    "remove_outliers",
    "clip_outliers",
    "create_date_features",
    "transform_numeric",
}


def _fill_frame(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            X[col] = series.fillna(series.median())
        else:
            mode = series.mode(dropna=True)
            X[col] = series.fillna(mode.iloc[0] if not mode.empty else "missing")
    return X


def prepare_xy(
    df: pd.DataFrame,
    target: str,
    task: Task,
    dataset: str = "",
) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target column not found: {target!r}. Columns: {list(df.columns)}")

    df = df.copy()
    if dataset == "titanic" or "titanic" in str(dataset).lower():
        drop_cols = [c for c in TITANIC_DROP if c in df.columns]
        df = df.drop(columns=drop_cols)

    y = df[target]
    X = df.drop(columns=[target])
    X = X.dropna(axis=1, how="all")
    X = _fill_frame(X)

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    if not pd.api.types.is_numeric_dtype(y):
        y = pd.factorize(y)[0]
    y = y.astype(int) if task == "classification" else y.astype(float)
    return X, y


def apply_operations(
    df: pd.DataFrame, operations: list[dict[str, Any]]
) -> tuple[Preprocessor, list[dict[str, Any]]]:
    pp = Preprocessor(PreprocessorConfig(df=df.copy()))
    steps: list[dict[str, Any]] = []
    for raw in operations:
        if not isinstance(raw, dict):
            raise ValueError("Each operation must be an object.")
        op = raw.get("operation")
        if op not in PREPROCESSOR_METHODS:
            raise ValueError(
                f"Unknown operation: {op!r}. Available: {sorted(PREPROCESSOR_METHODS)}"
            )
        kwargs = {k: v for k, v in raw.items() if k != "operation"}
        getattr(pp, op)(**kwargs)
        rows, cols = pp.df.shape
        steps.append({"operation": op, "args": kwargs, "rows": rows, "cols": cols})
    return pp, steps


def preprocess_dataset(req: PreprocessRequest) -> dict[str, Any]:
    ref = resolve_ref(req.source, req.identifier, req.split, req.dataset)
    df = load_frame(ref)
    before = df.shape
    pp, steps = apply_operations(df, req.operations)
    after = pp.df.shape
    payload = preview_frame(
        pp.df,
        dataset_label(ref) if req.dataset is None else req.dataset,
        n_rows=req.n_rows,
        target=req.target,
    )
    payload["operations"] = pp.get_operations()
    payload["steps"] = steps
    payload["before_shape"] = before
    payload["after_shape"] = after
    return payload
