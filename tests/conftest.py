import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "salary": [50000, 60000, 70000, 80000, 90000],
            "city": ["NYC", "LA", "NYC", "SF", "LA"],
            "bought": [0, 1, 0, 1, 1],
        }
    )


@pytest.fixture
def df_with_missing():
    return pd.DataFrame(
        {
            "a": [1, None, 3, None, 5],
            "b": [None, 2, None, 4, None],
            "c": ["x", "y", None, "x", "y"],
        }
    )


@pytest.fixture
def csv_file(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def json_file(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    path = tmp_path / "test.json"
    df.to_json(path, orient="records")
    return path


@pytest.fixture
def parquet_file(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    path = tmp_path / "test.parquet"
    df.to_parquet(path, index=False)
    return path
