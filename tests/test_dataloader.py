import pandas as pd
import pytest

from data.dataloader import READERS, SKLEARN_DATASETS, DataLoader, Dataset


class TestDatasetDataclass:
    def test_default_values(self):
        ds = Dataset(source="local", identifier="data.csv")
        assert ds.source == "local"
        assert ds.identifier == "data.csv"
        assert ds.split is None
        assert ds.as_frame is True
        assert ds.kwargs == {}

    def test_custom_kwargs(self):
        ds = Dataset(source="local", identifier="data.csv", kwargs={"sep": ";"})
        assert ds.kwargs == {"sep": ";"}

    def test_sklearn_source(self):
        ds = Dataset(source="sklearn", identifier="iris")
        assert ds.source == "sklearn"


class TestDataLoaderRepr:
    def test_repr(self):
        loader = DataLoader(Dataset(source="local", identifier="data.csv"))
        assert repr(loader) == "DataLoader(source='local', identifier='data.csv')"


class TestLocalLoading:
    def test_csv(self, csv_file):
        loader = DataLoader(Dataset(source="local", identifier=csv_file))
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["x", "y"]
        assert len(df) == 3

    def test_json(self, json_file):
        loader = DataLoader(Dataset(source="local", identifier=json_file))
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_parquet(self, parquet_file):
        loader = DataLoader(Dataset(source="local", identifier=parquet_file))
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_file_not_found(self, tmp_path):
        loader = DataLoader(Dataset(source="local", identifier=tmp_path / "nope.csv"))
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            loader.load()

    def test_directory_not_file(self, tmp_path):
        loader = DataLoader(Dataset(source="local", identifier=tmp_path))
        with pytest.raises(ValueError, match="not a file"):
            loader.load()

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "data.xyz"
        path.write_text("hello")
        loader = DataLoader(Dataset(source="local", identifier=path))
        with pytest.raises(ValueError, match="Unsupported dataset format"):
            loader.load()

    def test_unsupported_source(self):
        loader = DataLoader(Dataset(source="bad", identifier="x"))
        with pytest.raises(ValueError, match="Unsupported dataset source"):
            loader.load()


class TestSklearnLoading:
    def test_iris(self):
        loader = DataLoader(Dataset(source="sklearn", identifier="iris"))
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 150
        assert "target" in df.columns
        assert "sepal length (cm)" in df.columns

    def test_wine(self):
        loader = DataLoader(Dataset(source="sklearn", identifier="wine"))
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 178
        assert "target" in df.columns

    def test_breast_cancer(self):
        loader = DataLoader(Dataset(source="sklearn", identifier="breast_cancer"))
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 569

    def test_diabetes(self):
        loader = DataLoader(Dataset(source="sklearn", identifier="diabetes"))
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 442
        assert "target" in df.columns

    def test_digits(self):
        loader = DataLoader(Dataset(source="sklearn", identifier="digits"))
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1797

    def test_linnerud(self):
        loader = DataLoader(Dataset(source="sklearn", identifier="linnerud"))
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20

    def test_unknown_dataset(self):
        loader = DataLoader(Dataset(source="sklearn", identifier="nonexistent"))
        with pytest.raises(ValueError, match="Unknown sklearn dataset"):
            loader.load()

    def test_case_insensitive(self):
        loader = DataLoader(Dataset(source="sklearn", identifier="IRIS"))
        df = loader.load()
        assert len(df) == 150


class TestReadersCoverage:
    def test_readers_dict_keys(self):
        expected = {
            ".csv",
            ".json",
            ".xlsx",
            ".xls",
            ".xlsm",
            ".parquet",
            ".feather",
            ".xml",
            ".sas7bdat",
            ".xpt",
            ".sav",
            ".dta",
        }
        assert set(READERS.keys()) == expected

    def test_sklearn_datasets_dict_keys(self):
        expected = {"iris", "wine", "breast_cancer", "diabetes", "digits", "linnerud"}
        assert set(SKLEARN_DATASETS.keys()) == expected
