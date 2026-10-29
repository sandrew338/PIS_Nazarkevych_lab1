import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_preparation import load_data, prepare_features

DATA_PATH = "data/processed/churn_v1.csv"
DATA_VERSION = "v1"
EXPERIMENT_NAME = "customer-churn-baseline"
REGISTERED_MODEL_NAME = "CustomerChurnClassifier"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def build_model(model_type: str, n_estimators: int = 100, max_iter: int = 2000):
    """Повертає модель і словник її гіперпараметрів для логування."""
    if model_type == "random_forest":
        params = {"n_estimators": n_estimators}
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE)
        return model, params
    params = {"max_iter": max_iter, "scaler": "StandardScaler"}
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=max_iter)),
        ]
    )
    return model, params


def main(model_type="logistic_regression", n_estimators=100, max_iter=2000):
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(DATA_PATH)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model, model_params = build_model(model_type, n_estimators, max_iter)

    with mlflow.start_run(run_name=f"{model_type}_{DATA_VERSION}"):
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("data_version", DATA_VERSION)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        for name, value in model_params.items():
            mlflow.log_param(name, value)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        Path("models").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        joblib.dump(model, "models/latest_model.joblib")

        # Список колонок потрібен inference-сервісу, щоб зібрати ті самі ознаки, що й при навчанні
        with open("models/feature_columns.json", "w", encoding="utf-8") as f:
            json.dump(list(X.columns), f, ensure_ascii=False, indent=2)

        report_path = "reports/latest_metrics.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {"model_type": model_type, "data_version": DATA_VERSION, **metrics},
                f,
                indent=2,
            )

        mlflow.log_artifact(report_path)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            # модель навчена нами самими, тому тип дерев випадкового лісу можна довірити
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print({"model_type": model_type, "data_version": DATA_VERSION, **metrics})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Навчання моделі з логуванням у MLflow")
    parser.add_argument(
        "--model-type",
        choices=["logistic_regression", "random_forest"],
        default="logistic_regression",
    )
    parser.add_argument("--n-estimators", type=int, default=100, help="для random_forest")
    parser.add_argument("--max-iter", type=int, default=2000, help="для logistic_regression")
    args = parser.parse_args()
    main(args.model_type, args.n_estimators, args.max_iter)
