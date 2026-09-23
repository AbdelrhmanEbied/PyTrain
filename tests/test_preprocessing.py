import numpy as np
import pandas as pd
import pytest

from data.preprocessing import Preprocessor, PreprocessorConfig, SplitResult


def make_pp(df):
    return Preprocessor(PreprocessorConfig(df=df.copy()))


class TestPreprocessorRepr:
    def test_repr(self, sample_df):
        pp = make_pp(sample_df)
        r = repr(pp)
        assert "rows=5" in r
        assert "cols=4" in r
        assert "operations=0" in r


class TestPreprocessorEq:
    def test_equal(self, sample_df):
        a = make_pp(sample_df)
        b = make_pp(sample_df)
        assert a == b

    def test_not_equal_different_data(self, sample_df):
        a = make_pp(sample_df)
        b = make_pp(sample_df.drop(index=0))
        assert a != b

    def test_not_equal_different_ops(self, sample_df):
        a = make_pp(sample_df)
        b = make_pp(sample_df)
        b.operations.append({"op": "test"})
        assert a != b

    def test_not_equal_other_type(self, sample_df):
        a = make_pp(sample_df)
        assert a != "not a preprocessor"


class TestClone:
    def test_clone_independence(self, sample_df):
        pp = make_pp(sample_df)
        cloned = pp.clone()
        assert pp == cloned
        cloned.drop_columns("age")
        assert pp.shape() == (5, 4)
        assert cloned.shape() == (5, 3)


class TestShapeAndColumns:
    def test_shape(self, sample_df):
        pp = make_pp(sample_df)
        assert pp.shape() == (5, 4)

    def test_columns(self, sample_df):
        pp = make_pp(sample_df)
        assert pp.columns() == ["age", "salary", "city", "bought"]


class TestInfo:
    def test_info(self, sample_df):
        pp = make_pp(sample_df)
        info = pp.info()
        assert isinstance(info, pd.DataFrame)
        assert len(info) == 4
        assert "dtype" in info.columns
        assert "missing" in info.columns


class TestDescribe:
    def test_describe(self, sample_df):
        pp = make_pp(sample_df)
        desc = pp.describe()
        assert isinstance(desc, pd.DataFrame)
        assert "mean" in desc.columns


class TestMissingValues:
    def test_no_missing(self, sample_df):
        pp = make_pp(sample_df)
        mv = pp.missing_values()
        assert len(mv) == 0

    def test_has_missing(self, df_with_missing):
        pp = make_pp(df_with_missing)
        mv = pp.missing_values()
        assert len(mv) == 3
        assert mv["missing"].iloc[0] == 3


class TestDuplicates:
    def test_no_duplicates(self, sample_df):
        pp = make_pp(sample_df)
        assert pp.duplicates() == 0

    def test_has_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        pp = make_pp(df)
        assert pp.duplicates() == 1


class TestUniqueValues:
    def test_unique_values(self, sample_df):
        pp = make_pp(sample_df)
        uv = pp.unique_values("city")
        assert isinstance(uv, pd.Series)
        assert uv.sum() == 5

    def test_missing_column(self, sample_df):
        pp = make_pp(sample_df)
        with pytest.raises(KeyError):
            pp.unique_values("nonexistent")


class TestClassDistribution:
    def test_class_distribution(self, sample_df):
        pp = make_pp(sample_df)
        cd = pp.class_distribution("bought")
        assert isinstance(cd, pd.Series)
        assert abs(cd.sum() - 100) < 0.1


class TestDropColumns:
    def test_drop_single(self, sample_df):
        pp = make_pp(sample_df)
        pp.drop_columns("city")
        assert "city" not in pp.df.columns
        assert pp.shape() == (5, 3)

    def test_drop_multiple(self, sample_df):
        pp = make_pp(sample_df)
        pp.drop_columns(["age", "salary"])
        assert pp.shape() == (5, 2)

    def test_drop_records_operation(self, sample_df):
        pp = make_pp(sample_df)
        pp.drop_columns("city")
        assert len(pp.operations) == 1
        assert pp.operations[0]["operation"] == "drop_columns"

    def test_drop_missing_column(self, sample_df):
        pp = make_pp(sample_df)
        with pytest.raises(KeyError):
            pp.drop_columns("nonexistent")


class TestRenameColumns:
    def test_rename(self, sample_df):
        pp = make_pp(sample_df)
        pp.rename_columns({"city": "location"})
        assert "location" in pp.df.columns
        assert "city" not in pp.df.columns

    def test_rename_records(self, sample_df):
        pp = make_pp(sample_df)
        pp.rename_columns({"city": "location"})
        assert pp.operations[0]["mapping"] == {"city": "location"}


class TestDropDuplicates:
    def test_drop_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        pp = make_pp(df)
        pp.drop_duplicates()
        assert len(pp.df) == 2

    def test_drop_duplicates_subset(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
        pp = make_pp(df)
        pp.drop_duplicates(subset=["a"])
        assert len(pp.df) == 2


class TestDropMissingRows:
    def test_drop_any(self, df_with_missing):
        pp = make_pp(df_with_missing)
        pp.drop_missing_rows(how="any")
        assert len(pp.df) == 0

    def test_drop_all(self, df_with_missing):
        pp = make_pp(df_with_missing)
        pp.drop_missing_rows(how="all")
        assert len(pp.df) == 5

    def test_drop_specific_columns(self, df_with_missing):
        pp = make_pp(df_with_missing)
        pp.drop_missing_rows(columns=["a"], how="any")
        assert len(pp.df) == 3


class TestFillMissing:
    def test_fill_median(self, df_with_missing):
        pp = make_pp(df_with_missing)
        pp.fill_missing(["a", "b"], strategy="median")
        assert pp.df["a"].isna().sum() == 0
        assert pp.df["b"].isna().sum() == 0

    def test_fill_mean(self, df_with_missing):
        pp = make_pp(df_with_missing)
        pp.fill_missing(["a", "b"], strategy="mean")
        assert pp.df["a"].isna().sum() == 0

    def test_fill_most_frequent(self, df_with_missing):
        pp = make_pp(df_with_missing)
        pp.fill_missing(["c"], strategy="most_frequent")
        assert pp.df["c"].isna().sum() == 0

    def test_fill_constant(self, df_with_missing):
        pp = make_pp(df_with_missing)
        pp.fill_missing(["c"], strategy="constant", fill_value="N/A")
        assert pp.df["c"].isna().sum() == 0
        assert (pp.df["c"] == "N/A").sum() == 1

    def test_fill_mean_non_numeric_raises(self, df_with_missing):
        pp = make_pp(df_with_missing)
        with pytest.raises(TypeError, match="can only be used with numeric"):
            pp.fill_missing(["c"], strategy="mean")

    def test_fill_missing_column(self, sample_df):
        pp = make_pp(sample_df)
        with pytest.raises(KeyError):
            pp.fill_missing(["nonexistent"])


class TestConvertDtype:
    def test_to_numeric(self):
        df = pd.DataFrame({"a": ["1", "2", "3"]})
        pp = make_pp(df)
        pp.convert_dtype("a", "numeric")
        assert pd.api.types.is_numeric_dtype(pp.df["a"])

    def test_to_string(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        pp = make_pp(df)
        pp.convert_dtype("a", "string")
        assert str(pp.df["a"].dtype) == "string"

    def test_to_category(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        pp = make_pp(df)
        pp.convert_dtype("a", "category")
        assert pp.df["a"].dtype.name == "category"

    def test_unsupported_dtype(self, sample_df):
        pp = make_pp(sample_df)
        with pytest.raises(ValueError, match="Unsupported dtype"):
            pp.convert_dtype("age", "complex")


class TestParseDates:
    def test_parse_dates(self):
        df = pd.DataFrame({"date": ["2024-01-01", "2024-06-15", "2024-12-31"]})
        pp = make_pp(df)
        pp.parse_dates("date")
        assert pd.api.types.is_datetime64_any_dtype(pp.df["date"])


class TestCleanNumeric:
    def test_clean_currency(self):
        df = pd.DataFrame({"price": ["$1,000", "$2,500", "$3,000"]})
        pp = make_pp(df)
        pp.clean_numeric("price")
        assert pp.df["price"].iloc[0] == 1000.0

    def test_clean_percent(self):
        df = pd.DataFrame({"rate": ["50%", "75.5%", "100%"]})
        pp = make_pp(df)
        pp.clean_numeric("rate", percent=True)
        assert pp.df["rate"].iloc[0] == pytest.approx(0.5)

    def test_clean_euro(self):
        df = pd.DataFrame({"price": ["€100", "€200"]})
        pp = make_pp(df)
        pp.clean_numeric("price", currency_symbols="€")
        assert pp.df["price"].iloc[0] == 100.0


class TestFilterRows:
    def test_filter(self, sample_df):
        pp = make_pp(sample_df)
        pp.filter_rows("age > 30")
        assert len(pp.df) == 3

    def test_filter_invalid(self, sample_df):
        pp = make_pp(sample_df)
        with pytest.raises(ValueError, match="Invalid row filter"):
            pp.filter_rows("invalid !!!")


class TestShuffle:
    def test_shuffle(self, sample_df):
        pp = make_pp(sample_df)
        pp.shuffle(random_state=0)
        assert len(pp.df) == 5


class TestSampleRows:
    def test_sample_n(self, sample_df):
        pp = make_pp(sample_df)
        pp.sample_rows(n=3, random_state=0)
        assert len(pp.df) == 3

    def test_sample_frac(self, sample_df):
        pp = make_pp(sample_df)
        pp.sample_rows(frac=0.6, random_state=0)
        assert len(pp.df) == 3

    def test_sample_neither(self, sample_df):
        pp = make_pp(sample_df)
        with pytest.raises(ValueError, match="Provide either"):
            pp.sample_rows()

    def test_sample_both(self, sample_df):
        pp = make_pp(sample_df)
        with pytest.raises(ValueError, match="not both"):
            pp.sample_rows(n=2, frac=0.5)


class TestComputeBounds:
    def test_iqr(self):
        pp = make_pp(pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]}))
        lower, upper = pp._compute_bounds(pp.df["x"], "iqr", 1.5)
        assert lower < 1
        assert upper > 5

    def test_zscore(self):
        pp = make_pp(pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]}))
        lower, upper = pp._compute_bounds(pp.df["x"], "zscore", 2.0)
        assert lower < 1
        assert upper > 5

    def test_zscore_zero_std(self):
        pp = make_pp(pd.DataFrame({"x": [5, 5, 5]}))
        lower, upper = pp._compute_bounds(pp.df["x"], "zscore", 2.0)
        assert lower == float("-inf")
        assert upper == float("inf")

    def test_invalid_method(self):
        pp = make_pp(pd.DataFrame({"x": [1, 2, 3]}))
        with pytest.raises(ValueError, match="method must be"):
            pp._compute_bounds(pp.df["x"], "bad", 1.5)


class TestDetectOutliers:
    def test_iqr(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        pp = make_pp(df)
        outliers = pp.detect_outliers("x", "iqr", 1.5)
        assert outliers["x"].sum() == 1

    def test_zscore(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        pp = make_pp(df)
        outliers = pp.detect_outliers("x", "zscore", 2.0)
        assert bool(outliers["x"].iloc[-1])

    def test_non_numeric_raises(self):
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        pp = make_pp(df)
        with pytest.raises(TypeError, match="requires numeric"):
            pp.detect_outliers("x")


class TestRemoveOutliers:
    def test_remove(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        pp = make_pp(df)
        pp.remove_outliers("x", "iqr", 1.5)
        assert len(pp.df) == 5


class TestClipOutliers:
    def test_clip(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        pp = make_pp(df)
        pp.clip_outliers("x", "iqr", 1.5)
        assert pp.df["x"].max() < 100

    def test_clip_non_numeric_raises(self):
        df = pd.DataFrame({"x": ["a", "b"]})
        pp = make_pp(df)
        with pytest.raises(TypeError, match="requires numeric"):
            pp.clip_outliers("x")


class TestCreateDateFeatures:
    def test_year_month_day(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-03-15", "2024-07-20"])})
        pp = make_pp(df)
        pp.create_date_features("date", ["year", "month", "day"])
        assert "date_year" in pp.df.columns
        assert "date_month" in pp.df.columns
        assert "date_day" in pp.df.columns
        assert pp.df["date_year"].iloc[0] == 2024
        assert pp.df["date_month"].iloc[0] == 3

    def test_day_of_week(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-03-18"])})
        pp = make_pp(df)
        pp.create_date_features("date", ["day_of_week"])
        assert pp.df["date_day_of_week"].iloc[0] == 0

    def test_day_of_year(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-03-15"])})
        pp = make_pp(df)
        pp.create_date_features("date", ["day_of_year"])
        assert pp.df["date_day_of_year"].iloc[0] == 75

    def test_week_of_year(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-03-15"])})
        pp = make_pp(df)
        pp.create_date_features("date", ["week_of_year"])
        assert "date_week_of_year" in pp.df.columns

    def test_quarter(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-03-15"])})
        pp = make_pp(df)
        pp.create_date_features("date", ["quarter"])
        assert pp.df["date_quarter"].iloc[0] == 1

    def test_hour(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-03-15 14:30:00"])})
        pp = make_pp(df)
        pp.create_date_features("date", ["hour"])
        assert pp.df["date_hour"].iloc[0] == 14

    def test_is_month_start(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-03-01", "2024-03-15"])})
        pp = make_pp(df)
        pp.create_date_features("date", ["is_month_start"])
        assert pp.df["date_is_month_start"].iloc[0] == 1
        assert pp.df["date_is_month_start"].iloc[1] == 0

    def test_is_month_end(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-02-29", "2024-03-15"])})
        pp = make_pp(df)
        pp.create_date_features("date", ["is_month_end"])
        assert pp.df["date_is_month_end"].iloc[0] == 1

    def test_is_weekend(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-03-16", "2024-03-18"])})
        pp = make_pp(df)
        pp.create_date_features("date", ["is_weekend"])
        assert pp.df["date_is_weekend"].iloc[0] == 1
        assert pp.df["date_is_weekend"].iloc[1] == 0

    def test_unsupported_feature(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-03-15"])})
        pp = make_pp(df)
        with pytest.raises(ValueError, match="Unsupported date feature"):
            pp.create_date_features("date", ["invalid_feature"])


class TestTransformNumeric:
    def test_log1p(self):
        df = pd.DataFrame({"x": [0, 1, 2, 3]})
        pp = make_pp(df)
        pp.transform_numeric("x", "log1p")
        expected = np.log1p(df["x"])
        pd.testing.assert_series_equal(pp.df["x"], expected, check_names=False)

    def test_sqrt(self):
        df = pd.DataFrame({"x": [0, 1, 4, 9]})
        pp = make_pp(df)
        pp.transform_numeric("x", "sqrt")
        expected = np.sqrt(df["x"])
        pd.testing.assert_series_equal(pp.df["x"], expected, check_names=False)

    def test_square(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        pp = make_pp(df)
        pp.transform_numeric("x", "square")
        expected = np.square(df["x"])
        pd.testing.assert_series_equal(pp.df["x"], expected, check_names=False)

    def test_absolute(self):
        df = pd.DataFrame({"x": [-1, 2, -3]})
        pp = make_pp(df)
        pp.transform_numeric("x", "absolute")
        expected = np.abs(df["x"])
        pd.testing.assert_series_equal(pp.df["x"], expected, check_names=False)

    def test_log1p_negative_raises(self):
        df = pd.DataFrame({"x": [-2, 1, 2]})
        pp = make_pp(df)
        with pytest.raises(ValueError, match="below -1"):
            pp.transform_numeric("x", "log1p")

    def test_sqrt_negative_raises(self):
        df = pd.DataFrame({"x": [-1, 1, 2]})
        pp = make_pp(df)
        with pytest.raises(ValueError, match="negative"):
            pp.transform_numeric("x", "sqrt")

    def test_non_numeric_raises(self):
        df = pd.DataFrame({"x": ["a", "b"]})
        pp = make_pp(df)
        with pytest.raises(TypeError, match="requires numeric"):
            pp.transform_numeric("x", "log1p")


class TestSplit:
    def test_train_test(self, sample_df):
        pp = make_pp(sample_df)
        result = pp.split("bought", test_size=0.4, random_state=42)
        assert isinstance(result, SplitResult)
        assert result.X_train is not None
        assert result.X_validation is None
        total = len(result.X_train) + len(result.X_test)
        assert total == 5

    def test_train_val_test(self, sample_df):
        pp = make_pp(sample_df)
        result = pp.split("bought", test_size=0.2, validation_size=0.2, random_state=42)
        assert result.X_validation is not None
        total = len(result.X_train) + len(result.X_validation) + len(result.X_test)
        assert total == 5

    def test_stratify(self, sample_df):
        pp = make_pp(sample_df)
        result = pp.split("bought", test_size=0.4, stratify=True, random_state=42)
        assert len(result.X_train) + len(result.X_test) == 5

    def test_validation_too_large(self, sample_df):
        pp = make_pp(sample_df)
        with pytest.raises(ValueError, match="must be less than 1"):
            pp.split("bought", test_size=0.5, validation_size=0.6)

    def test_test_too_large(self, sample_df):
        pp = make_pp(sample_df)
        with pytest.raises(ValueError):
            pp.split("bought", test_size=1.5)

    def test_sum_too_large(self, sample_df):
        pp = make_pp(sample_df)
        with pytest.raises(ValueError, match="less than 1"):
            pp.split("bought", test_size=0.6, validation_size=0.5)

    def test_missing_target(self, sample_df):
        pp = make_pp(sample_df)
        with pytest.raises(KeyError):
            pp.split("nonexistent")


class TestBuildColumnTransformer:
    def test_numeric_only(self, sample_df):
        pp = make_pp(sample_df)
        ct = pp.build_column_transformer(["age", "salary"], [])
        assert ct is not None

    def test_categorical_only(self, sample_df):
        pp = make_pp(sample_df)
        ct = pp.build_column_transformer([], ["city"])
        assert ct is not None

    def test_both(self, sample_df):
        pp = make_pp(sample_df)
        ct = pp.build_column_transformer(["age", "salary"], ["city"])
        assert ct is not None

    def test_no_columns_raises(self, sample_df):
        pp = make_pp(sample_df)
        with pytest.raises(ValueError, match="At least one"):
            pp.build_column_transformer([], [])

    def test_scalings(self, sample_df):
        pp = make_pp(sample_df)
        for scaling in ["standard", "minmax", "robust", "maxabs", "none"]:
            ct = pp.build_column_transformer(["age"], [], scaling=scaling)
            assert ct is not None

    def test_encodings(self, sample_df):
        pp = make_pp(sample_df)
        for enc in ["one_hot", "ordinal"]:
            ct = pp.build_column_transformer([], ["city"], categorical_encoding=enc)
            assert ct is not None

    def test_remainder(self, sample_df):
        pp = make_pp(sample_df)
        ct = pp.build_column_transformer(["age"], ["city"], remainder="passthrough")
        assert ct.remainder == "passthrough"

    def test_invalid_encoding(self, sample_df):
        pp = make_pp(sample_df)
        with pytest.raises(ValueError, match="categorical_encoding"):
            pp.build_column_transformer([], ["city"], categorical_encoding="bad")


class TestFitTransformAndTransform:
    def test_fit_transform(self, sample_df):
        pp = make_pp(sample_df)
        ct = pp.build_column_transformer(["age", "salary"], ["city"])
        result = pp.fit_transform(ct, pp.df[["age", "salary", "city"]])
        assert result.shape[0] == 5

    def test_transform(self, sample_df):
        pp = make_pp(sample_df)
        ct = pp.build_column_transformer(["age", "salary"], ["city"])
        pp.fit_transform(ct, pp.df[["age", "salary", "city"]])
        result = pp.transform(ct, pp.df[["age", "salary", "city"]])
        assert result.shape[0] == 5


class TestFeatureSelection:
    def test_variance_threshold(self):
        vt = Preprocessor.variance_threshold(0.0)
        assert vt is not None

    def test_select_k_best_classification(self):
        skb = Preprocessor.select_k_best(k=2, task="classification")
        assert skb is not None

    def test_select_k_best_regression(self):
        skb = Preprocessor.select_k_best(k=2, task="regression")
        assert skb is not None

    def test_polynomial_features(self):
        pf = Preprocessor.polynomial_features(degree=2)
        assert pf is not None


class TestGetDataAndOperations:
    def test_get_data_returns_copy(self, sample_df):
        pp = make_pp(sample_df)
        data = pp.get_data()
        data.drop(columns=["age"], inplace=True)
        assert "age" in pp.df.columns

    def test_get_operations(self, sample_df):
        pp = make_pp(sample_df)
        pp.drop_columns("city")
        ops = pp.get_operations()
        assert len(ops) == 1
        ops[0]["op"] = "modified"
        assert pp.operations[0]["operation"] == "drop_columns"


class TestRequireColumns:
    def test_valid_columns(self, sample_df):
        pp = make_pp(sample_df)
        pp._require_columns(["age", "salary"])

    def test_missing_columns(self, sample_df):
        pp = make_pp(sample_df)
        with pytest.raises(KeyError, match="not found"):
            pp._require_columns(["nonexistent"])


class TestAsList:
    def test_string(self):
        assert Preprocessor._as_list("a") == ["a"]

    def test_list(self):
        assert Preprocessor._as_list(["a", "b"]) == ["a", "b"]

    def test_tuple(self):
        assert Preprocessor._as_list(("a", "b")) == ["a", "b"]


class TestOperationRecording:
    def test_chained_operations(self, sample_df):
        pp = make_pp(sample_df)
        pp.drop_columns("city").rename_columns({"age": "years"}).shuffle()
        assert len(pp.operations) == 3
        assert [op["operation"] for op in pp.operations] == [
            "drop_columns",
            "rename_columns",
            "shuffle",
        ]
