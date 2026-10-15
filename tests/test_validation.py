import sys

import pandas as pd

sys.path.insert(0, "src")
from validate_data import load_schema, validate_data  # noqa: E402

SCHEMA = load_schema("configs/data_schema.yaml")


def test_raw_data_passes():
    df = pd.read_csv("data/raw/churn.csv")
    assert validate_data(df, SCHEMA)["status"] == "passed"


def test_missing_column_fails():
    df = pd.read_csv("data/raw/churn.csv").drop(columns=["tenure"])
    report = validate_data(df, SCHEMA)
    assert report["status"] == "failed"
    assert report["errors"][0]["type"] == "missing_columns"


def test_unexpected_target_value_fails():
    df = pd.read_csv("data/raw/churn.csv")
    df.loc[0, "Churn"] = "Maybe"
    report = validate_data(df, SCHEMA)
    assert any(e["type"] == "unexpected_target_values" for e in report["errors"])


def test_duplicates_are_warning_not_error():
    df = pd.read_csv("data/raw/churn.csv")
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    report = validate_data(df, SCHEMA)
    assert report["status"] == "passed"
    assert report["warnings"][0]["type"] == "duplicates"
