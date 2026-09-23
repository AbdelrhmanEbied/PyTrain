from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from datasets import load_dataset
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_linnerud,
    load_wine,
)

READERS = {
    ".csv": pd.read_csv,
    ".json": pd.read_json,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".xlsm": pd.read_excel,
    ".parquet": pd.read_parquet,
    ".feather": pd.read_feather,
    ".xml": pd.read_xml,
    ".sas7bdat": pd.read_sas,
    ".xpt": pd.read_sas,
    ".sav": pd.read_spss,
    ".dta": pd.read_stata,
}

SKLEARN_DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "diabetes": load_diabetes,
    "digits": load_digits,
    "linnerud": load_linnerud,
}


@dataclass
class Dataset:
    source: Literal["huggingface", "local", "sklearn"]
    identifier: str | Path
    split: str | None = None
    as_frame: bool = True
    kwargs: dict[str, Any] = field(default_factory=dict)


class DataLoader:
    def __init__(self, cfg: Dataset) -> None:
        self.source = cfg.source
        self.identifier = cfg.identifier
        self.split = cfg.split
        self.as_frame = cfg.as_frame
        self.kwargs = cfg.kwargs

    def __repr__(self) -> str:
        return f"DataLoader(source={self.source!r}, identifier={self.identifier!r})"

    def load(self) -> pd.DataFrame:
        if self.source == "huggingface":
            return self._load_huggingface()

        if self.source == "local":
            return self._load_local()

        if self.source == "sklearn":
            return self._load_sklearn()

        raise ValueError(f"Unsupported dataset source: {self.source}")

    def _load_local(self) -> pd.DataFrame:
        path = Path(self.identifier)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        if not path.is_file():
            raise ValueError(f"Dataset path is not a file: {path}")

        extension = path.suffix.lower()

        reader = READERS.get(extension)

        if reader is None:
            supported = ", ".join(sorted(READERS))
            raise ValueError(
                f"Unsupported dataset format: {extension}. Supported formats: {supported}"
            )

        return reader(path, **self.kwargs)

    def _load_huggingface(self) -> pd.DataFrame:
        dataset = load_dataset(str(self.identifier), **self.kwargs)

        if self.split is not None:
            if self.split not in dataset:
                available = ", ".join(dataset.keys())
                raise ValueError(f"Split {self.split!r} not found. Available: {available}")
            split_data = dataset[self.split]
            df = split_data.to_pandas()
            if isinstance(df, pd.DataFrame):
                return df
            return pd.concat(list(df), ignore_index=True)

        if "train" in dataset:
            train = dataset["train"]
            df = train.to_pandas()
            if isinstance(df, pd.DataFrame):
                return df
            return pd.concat(list(df), ignore_index=True)

        first_split = next(iter(dataset))
        split_data = dataset[first_split]
        df = split_data.to_pandas()
        if isinstance(df, pd.DataFrame):
            return df
        return pd.concat(list(df), ignore_index=True)

    def _load_sklearn(self) -> pd.DataFrame:
        name = str(self.identifier).lower()

        loader = SKLEARN_DATASETS.get(name)
        if loader is None:
            available = ", ".join(sorted(SKLEARN_DATASETS))
            raise ValueError(f"Unknown sklearn dataset: {name!r}. Available: {available}")

        bunch = loader(**self.kwargs)

        if self.as_frame and bunch.get("frame") is not None:
            return bunch["frame"]

        df = pd.DataFrame(bunch["data"], columns=bunch["feature_names"])

        if "target" in bunch:
            target = bunch["target"]
            if target.ndim == 1:
                if "target" not in df.columns:
                    df["target"] = target
            else:
                for i in range(target.shape[1]):
                    col_name = f"target_{i}"
                    if col_name not in df.columns:
                        df[col_name] = target[:, i]

        return df
