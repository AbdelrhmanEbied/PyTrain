export const OP_GROUPS = [
  {
    label: "Clean",
    ops: [
      "drop_columns",
      "rename_columns",
      "drop_duplicates",
      "drop_missing_rows",
      "fill_missing",
      "clean_numeric",
      "parse_dates",
      "convert_dtype",
    ],
  },
  { label: "Filter & sample", ops: ["filter_rows", "shuffle", "sample_rows"] },
  { label: "Outliers", ops: ["remove_outliers", "clip_outliers"] },
  { label: "Transform", ops: ["transform_numeric", "create_date_features"] },
];

export const DEFAULT_ARGS = {
  drop_columns: '{"columns": ["col1"]}',
  rename_columns: '{"mapping": {"old": "new"}}',
  drop_duplicates: "{}",
  drop_missing_rows: '{"how": "any"}',
  fill_missing: '{"columns": ["age"], "strategy": "median"}',
  convert_dtype: '{"column": "col", "dtype": "numeric"}',
  parse_dates: '{"columns": ["date"]}',
  clean_numeric: '{"column": "price"}',
  filter_rows: '{"expression": "age > 18"}',
  shuffle: '{"random_state": 42}',
  sample_rows: '{"n": 100}',
  remove_outliers: '{"columns": ["x"], "method": "iqr", "threshold": 1.5}',
  clip_outliers: '{"columns": ["x"], "method": "iqr", "threshold": 1.5}',
  create_date_features: '{"column": "date", "features": ["year", "month"]}',
  transform_numeric: '{"columns": ["x"], "transformation": "log1p"}',
};
