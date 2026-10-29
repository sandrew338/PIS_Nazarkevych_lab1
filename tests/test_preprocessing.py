import json
import sys

import pandas as pd

sys.path.insert(0, "src")
from app.preprocessing import to_model_input  # noqa: E402
from data_preparation import prepare_features  # noqa: E402


def test_inference_features_match_training():
    df = pd.read_csv("data/processed/churn_v1.csv")
    X, _ = prepare_features(df)
    columns = list(X.columns)
    for i in (0, 5, 100, 3000):
        row = df.drop(columns=["customerID", "Churn"]).iloc[i].to_dict()
        got = to_model_input(row, columns)
        assert list(got.columns) == columns
        assert (got.iloc[0].values == X.iloc[i].astype(float).values).all()
