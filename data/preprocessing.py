from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

Task = Literal["classification", "regression"]
DateFeature = Literal[
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "quarter",
    "hour",
    "is_month_start",
    "is_month_end",
    "is_weekend",
]


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_validation: pd.DataFrame | None = None
    y_validation: pd.Series | None = None


@dataclass
class PreprocessorConfig:
    df: pd.DataFrame
    operations: list[dict[str, Any]] = field(default_factory=list)


class Preprocessor:
    def __init__(self, cfg: PreprocessorConfig) -> None:
        self.df = cfg.df
        self.operations = cfg.operations

    def __repr__(self) -> str:
        rows, cols = self.df.shape
        ops = len(self.operations)
        return f"Preprocessor(rows={rows}, cols={cols}, operations={ops})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Preprocessor):
            return NotImplemented
        return self.df.equals(other.df) and self.operations == other.operations

    def clone(self) -> Preprocessor:
        return Preprocessor(
            PreprocessorConfig(
                df=self.df.copy(),
                operations=copy.deepcopy(self.operations),
            )
        )

    def shape(self) -> tuple[int, int]:
        return self.df.shape

    def columns(self) -> list[str]:
        return self.df.columns.tolist()

    def info(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "column": self.df.columns,
                "dtype": self.df.dtypes.astype(str).values,
                "non_null": self.df.notna().sum().values,
                "missing": self.df.isna().sum().values,
                "missing_pct": (self.df.isna().mean().mul(100).round(2).values),
                "unique": self.df.nunique(dropna=True).values,
            }
        )

    def describe(self, include: Any = "all") -> pd.DataFrame:
        return self.df.describe(include=include).T

    def missing_values(self) -> pd.DataFrame:
        result = pd.DataFrame(
            {
                "missing": self.df.isna().sum(),
                "missing_pct": self.df.isna().mean().mul(100).round(2),
            }
        )
        return result[result["missing"] > 0].sort_values("missing", ascending=False)

    def duplicates(self) -> int:
        return int(self.df.duplicated().sum())

    def unique_values(self, column: str) -> pd.Series:
        self._require_columns([column])
        return self.df[column].value_counts(dropna=False)

    def class_distribution(self, target: str) -> pd.Series:
        self._require_columns([target])
        return self.df[target].value_counts(normalize=True, dropna=False).mul(100).round(2)

    def drop_columns(self, columns: str | Sequence[str]) -> Preprocessor:
        columns = self._as_list(columns)
        self._require_columns(columns)
        self.df = self.df.drop(columns=columns)
        self._record("drop_columns", columns=columns)
        return self

    def rename_columns(self, mapping: dict[str, str]) -> Preprocessor:
        self._require_columns(list(mapping))
        self.df = self.df.rename(columns=mapping)
        self._record("rename_columns", mapping=mapping)
        return self

    def drop_duplicates(self, subset: Sequence[str] | None = None) -> Preprocessor:
        self.df = self.df.drop_duplicates(subset=subset).reset_index(drop=True)
        self._record("drop_duplicates", subset=subset)
        return self

    def drop_missing_rows(
        self,
        columns: Sequence[str] | None = None,
        how: Literal["any", "all"] = "any",
    ) -> Preprocessor:
        if columns is not None:
            self._require_columns(list(columns))
        self.df = self.df.dropna(subset=columns, how=how).reset_index(drop=True)
        self._record("drop_missing_rows", columns=columns, how=how)
        return self

    def fill_missing(
        self,
        columns: str | Sequence[str],
        strategy: Literal["mean", "median", "most_frequent", "constant"] = "median",
        fill_value: Any = "missing",
    ) -> Preprocessor:
        columns = self._as_list(columns)
        self._require_columns(columns)

        if strategy in {"mean", "median"}:
            non_numeric = [
                column for column in columns if not pd.api.types.is_numeric_dtype(self.df[column])
            ]
            if non_numeric:
                raise TypeError(
                    f"{strategy!r} can only be used with numeric columns: {non_numeric}"
                )

        if strategy == "constant":
            self.df[columns] = self.df[columns].fillna(fill_value)
        elif strategy == "mean":
            self.df[columns] = self.df[columns].fillna(self.df[columns].mean(numeric_only=True))
        elif strategy == "median":
            self.df[columns] = self.df[columns].fillna(self.df[columns].median(numeric_only=True))
        elif strategy == "most_frequent":
            for column in columns:
                mode = self.df[column].mode(dropna=True)
                if not mode.empty:
                    self.df[column] = self.df[column].fillna(mode.iloc[0])
        else:
            raise ValueError("strategy must be one of: mean, median, most_frequent, constant")

        self._record(
            "fill_missing",
            columns=columns,
            strategy=strategy,
            fill_value=fill_value,
        )
        return self

    def convert_dtype(
        self,
        column: str,
        dtype: str,
    ) -> Preprocessor:
        self._require_columns([column])

        if dtype == "numeric":
            self.df[column] = pd.to_numeric(self.df[column], errors="coerce")
        elif dtype == "string":
            self.df[column] = self.df[column].astype("string")
        elif dtype == "category":
            self.df[column] = self.df[column].astype("category")
        elif dtype in {"datetime", "datetime64[ns]"}:
            self.df[column] = pd.to_datetime(self.df[column], errors="coerce")
        elif dtype in {"int", "float", "bool"}:
            self.df[column] = self.df[column].astype(dtype)
        else:
            raise ValueError(
                "Unsupported dtype. Use numeric, string, category, datetime, int, float, or bool."
            )

        self._record("convert_dtype", column=column, dtype=dtype)
        return self

    def parse_dates(self, columns: str | Sequence[str]) -> Preprocessor:
        columns = self._as_list(columns)
        self._require_columns(columns)

        for column in columns:
            self.df[column] = pd.to_datetime(self.df[column], errors="coerce")

        self._record("parse_dates", columns=columns)
        return self

    def clean_numeric(
        self,
        column: str,
        percent: bool = False,
        currency_symbols: str = "$€£¥",
    ) -> Preprocessor:
        self._require_columns([column])

        series = self.df[column].astype("string").str.strip()

        if percent:
            series = series.str.replace("%", "", regex=False)
            numeric = pd.to_numeric(series, errors="coerce") / 100.0
        else:
            regex = "[" + re.escape(currency_symbols) + "]"
            series = series.str.replace(regex, "", regex=True)
            series = series.str.replace(",", "", regex=False)
            numeric = pd.to_numeric(series, errors="coerce")

        self.df[column] = numeric
        self._record(
            "clean_numeric",
            column=column,
            percent=percent,
            currency_symbols=currency_symbols,
        )
        return self

    def filter_rows(self, expression: str) -> Preprocessor:
        try:
            self.df = self.df.query(expression).reset_index(drop=True)
        except Exception as exc:
            raise ValueError(f"Invalid row filter: {expression!r}") from exc

        self._record("filter_rows", expression=expression)
        return self

    def shuffle(self, random_state: int = 42) -> Preprocessor:
        self.df = self.df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        self._record("shuffle", random_state=random_state)
        return self

    def sample_rows(
        self,
        n: int | None = None,
        frac: float | None = None,
        random_state: int = 42,
    ) -> Preprocessor:
        if n is None and frac is None:
            raise ValueError("Provide either n or frac.")

        if n is not None and frac is not None:
            raise ValueError("Provide n or frac, not both.")

        self.df = self.df.sample(
            n=n,
            frac=frac,
            random_state=random_state,
        ).reset_index(drop=True)

        self._record(
            "sample_rows",
            n=n,
            frac=frac,
            random_state=random_state,
        )
        return self

    def _compute_bounds(
        self,
        series: pd.Series,
        method: Literal["iqr", "zscore"],
        threshold: float,
    ) -> tuple[float, float]:
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            return q1 - threshold * iqr, q3 + threshold * iqr
        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            if std == 0 or pd.isna(std):
                return float("-inf"), float("inf")
            return mean - threshold * std, mean + threshold * std
        else:
            raise ValueError("method must be 'iqr' or 'zscore'.")

    def detect_outliers(
        self,
        columns: str | Sequence[str],
        method: Literal["iqr", "zscore"] = "iqr",
        threshold: float = 1.5,
    ) -> pd.DataFrame:
        columns = self._as_list(columns)
        self._require_columns(columns)

        result = pd.DataFrame(False, index=self.df.index, columns=columns)

        for column in columns:
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                raise TypeError(f"Outlier detection requires numeric columns: {column}")

            lower, upper = self._compute_bounds(self.df[column], method, threshold)
            result[column] = (self.df[column] < lower) | (self.df[column] > upper)

        return result

    def remove_outliers(
        self,
        columns: str | Sequence[str],
        method: Literal["iqr", "zscore"] = "iqr",
        threshold: float = 1.5,
    ) -> Preprocessor:
        outliers = self.detect_outliers(columns, method, threshold)
        self.df = self.df.loc[~outliers.any(axis=1)].reset_index(drop=True)
        self._record(
            "remove_outliers",
            columns=self._as_list(columns),
            method=method,
            threshold=threshold,
        )
        return self

    def clip_outliers(
        self,
        columns: str | Sequence[str],
        method: Literal["iqr", "zscore"] = "iqr",
        threshold: float = 1.5,
    ) -> Preprocessor:
        columns = self._as_list(columns)
        self._require_columns(columns)

        for column in columns:
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                raise TypeError(f"Outlier clipping requires numeric columns: {column}")

            lower, upper = self._compute_bounds(self.df[column], method, threshold)
            if lower == float("-inf") and upper == float("inf"):
                continue
            self.df[column] = self.df[column].clip(lower=lower, upper=upper)

        self._record(
            "clip_outliers",
            columns=columns,
            method=method,
            threshold=threshold,
        )
        return self

    def create_date_features(
        self,
        column: str,
        features: Sequence[DateFeature],
    ) -> Preprocessor:
        self._require_columns([column])

        if not pd.api.types.is_datetime64_any_dtype(self.df[column]):
            self.df[column] = pd.to_datetime(self.df[column], errors="coerce")

        dt = self.df[column].dt

        for feature in features:
            new_column = f"{column}_{feature}"

            if feature == "year":
                self.df[new_column] = dt.year
            elif feature == "month":
                self.df[new_column] = dt.month
            elif feature == "day":
                self.df[new_column] = dt.day
            elif feature == "day_of_week":
                self.df[new_column] = dt.dayofweek
            elif feature == "day_of_year":
                self.df[new_column] = dt.dayofyear
            elif feature == "week_of_year":
                self.df[new_column] = dt.isocalendar().week.astype(int)
            elif feature == "quarter":
                self.df[new_column] = dt.quarter
            elif feature == "hour":
                self.df[new_column] = dt.hour
            elif feature == "is_month_start":
                self.df[new_column] = dt.is_month_start.astype(int)
            elif feature == "is_month_end":
                self.df[new_column] = dt.is_month_end.astype(int)
            elif feature == "is_weekend":
                self.df[new_column] = (dt.dayofweek >= 5).astype(int)
            else:
                raise ValueError(f"Unsupported date feature: {feature}")

        self._record(
            "create_date_features",
            column=column,
            features=list(features),
        )
        return self

    def transform_numeric(
        self,
        columns: str | Sequence[str],
        transformation: Literal["log1p", "sqrt", "square", "absolute"],
    ) -> Preprocessor:
        columns = self._as_list(columns)
        self._require_columns(columns)

        for column in columns:
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                raise TypeError(f"Numeric transformation requires numeric columns: {column}")

            values = self.df[column]

            if transformation == "log1p":
                if (values.dropna() < -1).any():
                    raise ValueError(f"log1p cannot be applied to values below -1 in {column}.")
                self.df[column] = np.log1p(values)
            elif transformation == "sqrt":
                if (values.dropna() < 0).any():
                    raise ValueError(f"sqrt cannot be applied to negative values in {column}.")
                self.df[column] = np.sqrt(values)
            elif transformation == "square":
                self.df[column] = np.square(values)
            elif transformation == "absolute":
                self.df[column] = np.abs(values)
            else:
                raise ValueError(
                    "Unsupported transformation. Use log1p, sqrt, square, or absolute."
                )

        self._record(
            "transform_numeric",
            columns=columns,
            transformation=transformation,
        )
        return self

    def split(
        self,
        target: str,
        test_size: float = 0.2,
        validation_size: float | None = None,
        random_state: int = 42,
        stratify: bool = False,
    ) -> SplitResult:
        self._require_columns([target])

        X = self.df.drop(columns=[target])
        y = self.df[target]

        stratify_values = y if stratify else None

        if validation_size is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_values,
            )
            return SplitResult(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            )

        if not (0 < validation_size < 1):
            raise ValueError("validation_size must be between 0 and 1.")

        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1.")

        if validation_size + test_size >= 1:
            raise ValueError("validation_size + test_size must be less than 1.")

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=validation_size + test_size,
            random_state=random_state,
            stratify=stratify_values,
        )

        relative_test_size = test_size / (validation_size + test_size)

        temp_stratify = y_temp if stratify else None

        X_validation, X_test, y_validation, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_test_size,
            random_state=random_state,
            stratify=temp_stratify,
        )

        return SplitResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_validation=X_validation,
            y_validation=y_validation,
        )

    def build_column_transformer(
        self,
        numeric_columns: Sequence[str],
        categorical_columns: Sequence[str],
        numeric_imputation: Literal["mean", "median", "most_frequent", "constant"] = "median",
        categorical_imputation: Literal["most_frequent", "constant"] = "most_frequent",
        categorical_encoding: Literal["one_hot", "ordinal"] = "one_hot",
        scaling: Literal["standard", "minmax", "robust", "maxabs", "none"] = "standard",
        remainder: Literal["drop", "passthrough"] = "drop",
    ) -> ColumnTransformer:

        numeric_columns = list(numeric_columns)
        categorical_columns = list(categorical_columns)

        numeric_steps: list[tuple[str, Any]] = [
            (
                "imputer",
                SimpleImputer(strategy=numeric_imputation),
            )
        ]

        if scaling != "none":
            scalers = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler(),
                "robust": RobustScaler(),
                "maxabs": MaxAbsScaler(),
            }
            numeric_steps.append(("scaler", scalers[scaling]))

        categorical_steps: list[tuple[str, Any]] = [
            (
                "imputer",
                SimpleImputer(
                    strategy=categorical_imputation,
                    fill_value="missing",
                ),
            )
        ]

        if categorical_encoding == "one_hot":
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        elif categorical_encoding == "ordinal":
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        else:
            raise ValueError("categorical_encoding must be 'one_hot' or 'ordinal'.")

        categorical_steps.append(("encoder", encoder))

        transformers = []

        if numeric_columns:
            transformers.append(
                (
                    "numeric",
                    Pipeline(numeric_steps),
                    numeric_columns,
                )
            )

        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    Pipeline(categorical_steps),
                    categorical_columns,
                )
            )

        if not transformers:
            raise ValueError("At least one numeric or categorical column is required.")

        return ColumnTransformer(
            transformers=transformers,
            remainder=remainder,
            verbose_feature_names_out=False,
        )

    def fit_transform(
        self,
        transformer: ColumnTransformer | Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series | None = None,
    ) -> np.ndarray:
        return transformer.fit_transform(X_train, y_train)

    def transform(
        self,
        transformer: ColumnTransformer | Pipeline,
        X: pd.DataFrame,
    ) -> np.ndarray:
        return transformer.transform(X)

    @staticmethod
    def variance_threshold(
        threshold: float = 0.0,
    ) -> VarianceThreshold:
        return VarianceThreshold(threshold=threshold)

    @staticmethod
    def select_k_best(
        k: int,
        task: Task = "classification",
    ) -> SelectKBest:
        score_func = f_classif if task == "classification" else f_regression
        return SelectKBest(score_func=score_func, k=k)

    @staticmethod
    def polynomial_features(
        degree: int = 2,
        include_bias: bool = False,
    ) -> PolynomialFeatures:
        return PolynomialFeatures(
            degree=degree,
            include_bias=include_bias,
        )

    def get_data(self) -> pd.DataFrame:
        return self.df.copy()

    def get_operations(self) -> list[dict[str, Any]]:
        return [operation.copy() for operation in self.operations]

    def _record(self, operation: str, **kwargs: Any) -> None:
        self.operations.append(
            {
                "operation": operation,
                **kwargs,
            }
        )

    def _require_columns(self, columns: Sequence[str]) -> None:
        missing = [column for column in columns if column not in self.df.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")

    @staticmethod
    def _as_list(value: str | Sequence[str]) -> list[str]:
        if isinstance(value, str):
            return [value]
        return list(value)
