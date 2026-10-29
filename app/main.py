import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.preprocessing import to_model_input

MODEL_PATH = Path("models/latest_model.joblib")
COLUMNS_PATH = Path("models/feature_columns.json")

state: Dict[str, Any] = {"model": None, "columns": None}

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class ChurnRequest(BaseModel):
    gender: Literal["Female", "Male"] = "Female"
    SeniorCitizen: int = 0
    Partner: YesNo = "Yes"
    Dependents: YesNo = "No"
    tenure: int = 12
    PhoneService: YesNo = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: YesNoNoInternet = "No"
    OnlineBackup: YesNoNoInternet = "No"
    DeviceProtection: YesNoNoInternet = "No"
    TechSupport: YesNoNoInternet = "No"
    StreamingTV: YesNoNoInternet = "No"
    StreamingMovies: YesNoNoInternet = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: YesNo = "Yes"
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 850.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    for path in (MODEL_PATH, COLUMNS_PATH):
        if not path.exists():
            raise RuntimeError(f"File not found: {path}")
    state["model"] = joblib.load(MODEL_PATH)
    state["columns"] = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))
    yield
    state["model"] = None


app = FastAPI(title="Customer Churn Prediction API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.post("/predict")
def predict(payload: ChurnRequest) -> Dict[str, Any]:
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    X = to_model_input(payload.model_dump(), state["columns"])
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_label": "high" if probability >= 0.5 else "low",
    }
