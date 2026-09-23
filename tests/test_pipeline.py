from pathlib import Path

import pandas as pd
import pytest

from data.dataloader import DataLoader, Dataset
from data.preprocessing import Preprocessor, PreprocessorConfig
from mlflow_integration import MLflowTracker
from models import SklearnModel, XGBoostModel
from training import EvalResult, SklearnTrainer, TrainResult, XGBoostTrainer


@pytest.fixture(scope="module")
def titanic_csv():
    path = "data/test_datasets/titanic.csv"
    if not Path(path).exists():
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml("titanic", version=1, as_frame=True, parser="pandas")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        bunch.frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def raw_df(titanic_csv):
    loader = DataLoader(Dataset(source="local", identifier=titanic_csv))
    return loader.load()


@pytest.fixture(scope="module")
def cleaned_pp(raw_df):
    pp = Preprocessor(PreprocessorConfig(df=raw_df))
    pp.drop_columns(["boat", "body", "home.dest", "cabin", "ticket"])
    pp.rename_columns({"sibsp": "siblings", "parch": "parents_children"})
    pp.fill_missing(["age"], strategy="median")
    pp.fill_missing(["fare"], strategy="median")
    pp.fill_missing(["embarked"], strategy="most_frequent")
    pp.drop_duplicates()
    pp.convert_dtype("pclass", "category")
    pp.convert_dtype("sex", "category")
    pp.convert_dtype("embarked", "category")
    pp.clean_numeric("fare")
    pp.clean_numeric("age")
    pp.transform_numeric("fare", "log1p")
    pp.clip_outliers(["fare"], method="iqr", threshold=3.0)
    return pp


@pytest.fixture(scope="module")
def split_data(cleaned_pp):
    split = cleaned_pp.split(
        "survived",
        test_size=0.2,
        validation_size=0.1,
        stratify=True,
        random_state=42,
    )
    cat_cols = ["pclass", "sex", "embarked"]
    num_cols = ["age", "siblings", "parents_children", "fare"]

    X_train = pd.get_dummies(split.X_train[cat_cols + num_cols], columns=cat_cols, drop_first=True)
    X_val = pd.get_dummies(
        split.X_validation[cat_cols + num_cols], columns=cat_cols, drop_first=True
    )
    X_test = pd.get_dummies(split.X_test[cat_cols + num_cols], columns=cat_cols, drop_first=True)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return {
        "X_train": X_train.values,
        "y_train": split.y_train.astype(int).values,
        "X_val": X_val.values,
        "y_val": split.y_validation.astype(int).values,
        "X_test": X_test.values,
        "y_test": split.y_test.astype(int).values,
    }


class TestLoad:
    def test_loads_titanic(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)
        assert raw_df.shape == (1309, 14)
        assert "survived" in raw_df.columns


class TestInspect:
    def test_info(self, cleaned_pp):
        info = cleaned_pp.info()
        assert len(info) == 9

    def test_no_missing_after_clean(self, cleaned_pp):
        assert cleaned_pp.missing_values().shape[0] == 0

    def test_no_duplicates(self, cleaned_pp):
        assert cleaned_pp.duplicates() == 0


class TestClean:
    def test_dropped_columns(self, cleaned_pp):
        for col in ["boat", "body", "home.dest", "cabin", "ticket"]:
            assert col not in cleaned_pp.df.columns

    def test_renamed_columns(self, cleaned_pp):
        assert "siblings" in cleaned_pp.df.columns
        assert "parents_children" in cleaned_pp.df.columns

    def test_operations_recorded(self, cleaned_pp):
        ops = cleaned_pp.get_operations()
        assert len(ops) == 13


class TestSplit:
    def test_three_way_split(self, split_data):
        n_train = len(split_data["X_train"])
        n_val = len(split_data["X_val"])
        n_test = len(split_data["X_test"])
        assert n_train + n_val + n_test == 1309
        assert n_train > n_val
        assert n_train > n_test

    def test_aligned_columns(self, split_data):
        assert split_data["X_train"].shape[1] == split_data["X_val"].shape[1]
        assert split_data["X_train"].shape[1] == split_data["X_test"].shape[1]

    def test_binary_target(self, split_data):
        assert set(split_data["y_train"]) == {0, 1}


class TestSklearnTrain:
    def test_train_returns_result(self, split_data):
        model = SklearnModel("logistic_regression", "classification")
        model.set_params(max_iter=500)
        trainer = SklearnTrainer(model, "classification")
        result = trainer.train(
            split_data["X_train"],
            split_data["y_train"],
            X_val=split_data["X_val"],
            y_val=split_data["y_val"],
        )

        assert isinstance(result, TrainResult)
        assert "accuracy" in result.metrics
        assert result.train_time > 0
        assert len(result.predictions) == len(split_data["y_train"])
        assert result.model_info["model_name"] == "logistic_regression"

    def test_validation_metrics(self, split_data):
        model = SklearnModel("logistic_regression", "classification")
        model.set_params(max_iter=500)
        trainer = SklearnTrainer(model, "classification")
        result = trainer.train(
            split_data["X_train"],
            split_data["y_train"],
            X_val=split_data["X_val"],
            y_val=split_data["y_val"],
        )

        assert result.eval_result is not None
        assert "accuracy" in result.eval_result.metrics
        assert 0.0 <= result.eval_result.metrics["accuracy"] <= 1.0

    def test_evaluate_on_test(self, split_data):
        model = SklearnModel("logistic_regression", "classification")
        model.set_params(max_iter=500)
        trainer = SklearnTrainer(model, "classification")
        trainer.train(split_data["X_train"], split_data["y_train"])
        eval_result = trainer.evaluate(split_data["X_test"], split_data["y_test"])

        assert isinstance(eval_result, EvalResult)
        assert "accuracy" in eval_result.metrics
        assert eval_result.eval_time > 0
        assert len(eval_result.predictions) == len(split_data["y_test"])

    def test_accuracy_reasonable(self, split_data):
        model = SklearnModel("logistic_regression", "classification")
        model.set_params(max_iter=500)
        trainer = SklearnTrainer(model, "classification")
        trainer.train(split_data["X_train"], split_data["y_train"])
        eval_result = trainer.evaluate(split_data["X_test"], split_data["y_test"])

        assert eval_result.metrics["accuracy"] > 0.7


class TestXGBoostTrain:
    def test_train(self, split_data):
        model = XGBoostModel("xgb_classifier", "classification")
        model.set_params(n_estimators=50, max_depth=3, random_state=42)
        trainer = XGBoostTrainer(model, "classification")
        result = trainer.train(
            split_data["X_train"],
            split_data["y_train"],
            X_val=split_data["X_val"],
            y_val=split_data["y_val"],
        )

        assert isinstance(result, TrainResult)
        assert "accuracy" in result.metrics
        assert result.eval_result is not None

    def test_evaluate_on_test(self, split_data):
        model = XGBoostModel("xgb_classifier", "classification")
        model.set_params(n_estimators=50, max_depth=3, random_state=42)
        trainer = XGBoostTrainer(model, "classification")
        trainer.train(split_data["X_train"], split_data["y_train"])
        eval_result = trainer.evaluate(split_data["X_test"], split_data["y_test"])

        assert "accuracy" in eval_result.metrics
        assert eval_result.metrics["accuracy"] > 0.7


class TestMLflowLogging:
    def test_log_train_result(self, split_data, tmp_path):
        model = SklearnModel("logistic_regression", "classification")
        model.set_params(max_iter=500)
        trainer = SklearnTrainer(model, "classification")
        result = trainer.train(
            split_data["X_train"],
            split_data["y_train"],
            X_val=split_data["X_val"],
            y_val=split_data["y_val"],
        )

        tracker = MLflowTracker(
            experiment_name="titanic_pipeline_test",
            tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        )
        with tracker:
            tracker.log_train_result(result)

        assert result.metrics["accuracy"] > 0.7
        assert result.eval_result is not None
        assert "accuracy" in result.eval_result.metrics


class TestEndToEnd:
    def test_full_pipeline(self, raw_df):
        pp = Preprocessor(PreprocessorConfig(df=raw_df))
        pp.drop_columns(["boat", "body", "home.dest", "cabin", "ticket"])
        pp.rename_columns({"sibsp": "siblings", "parch": "parents_children"})
        pp.fill_missing(["age"], strategy="median")
        pp.fill_missing(["fare"], strategy="median")
        pp.fill_missing(["embarked"], strategy="most_frequent")
        pp.drop_duplicates()
        pp.convert_dtype("pclass", "category")
        pp.convert_dtype("sex", "category")
        pp.convert_dtype("embarked", "category")
        pp.clean_numeric("fare")
        pp.clean_numeric("age")
        pp.transform_numeric("fare", "log1p")
        pp.clip_outliers(["fare"], method="iqr", threshold=3.0)

        split = pp.split(
            "survived",
            test_size=0.2,
            validation_size=0.1,
            stratify=True,
            random_state=42,
        )

        cat_cols = ["pclass", "sex", "embarked"]
        num_cols = ["age", "siblings", "parents_children", "fare"]
        X_train = pd.get_dummies(
            split.X_train[cat_cols + num_cols], columns=cat_cols, drop_first=True
        )
        X_test = pd.get_dummies(
            split.X_test[cat_cols + num_cols], columns=cat_cols, drop_first=True
        )
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        model = SklearnModel("logistic_regression", "classification")
        model.set_params(max_iter=500)
        trainer = SklearnTrainer(model, "classification")
        result = trainer.train(X_train.values, split.y_train.astype(int).values)
        eval_result = trainer.evaluate(X_test.values, split.y_test.astype(int).values)

        assert isinstance(result, TrainResult)
        assert isinstance(eval_result, EvalResult)
        assert result.train_time > 0
        assert eval_result.metrics["accuracy"] > 0.7
        assert len(result.predictions) == len(split.y_train)
        assert len(eval_result.predictions) == len(split.y_test)
