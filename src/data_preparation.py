import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame, target_column: str = "Churn"):
    df = df.copy()

    # Ідентифікатор клієнта не несе сигналу і створив би ~7000 dummy-колонок
    df = df.drop(columns=["customerID"], errors="ignore")

    # TotalCharges у файлі - текст; порожні значення (' ') -> NaN -> 0
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    if not pd.api.types.is_numeric_dtype(y):
        y = y.map({"No": 0, "Yes": 1})

    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)
    return X, y.astype(int)
