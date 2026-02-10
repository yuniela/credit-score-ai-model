from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd
import time

from credit_scorer import CreditScorer

from collections import Counter

PREDICTION_COUNTER = Counter()


app = FastAPI(
    title="Credit Scoring API",
    version="1.0.0"
)

# Carga modelo al iniciar
scorer = CreditScorer(model_type="random_forest")
scorer.load("credit_model")
model_loaded_at = time.time()


# Pydantic Schemas
class CreditApplication(BaseModel):
    Age: int | None = None
    Annual_Income: float | None = None
    Monthly_Inhand_Salary: float | None = None
    Num_Bank_Accounts: int | None = None
    Num_Credit_Card: int | None = None
    Num_of_Loan: int | None = None
    Num_of_Delayed_Payment: int | None = None
    Outstanding_Debt: float | None = None
    Changed_Credit_Limit: float | None = None
    Credit_History_Age: str | None = None
    Payment_of_Min_Amount: str | None = None
    Payment_Behaviour: str | None = None
    Credit_Mix: str | None = None
    Delay_from_due_date: int | None = None
    Num_Credit_Inquiries: int | None = None
    Monthly_Balance: float | None = None

class PredictRequest(BaseModel):
    records: List[CreditApplication]

class PredictResponse(BaseModel):
    predictions: List[Any]
    probabilities: List[Dict[str, float]] | None = None
    model_type: str


# Endpoints

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": scorer.model is not None,
        "uptime_seconds": int(time.time() - model_loaded_at)
    }


@app.get("/model/info")
def model_info():
    if scorer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": scorer.model_type,
        "model_class": scorer.model.__class__.__name__,
        "num_features": len(scorer.fill_medians),
        "supports_proba": hasattr(scorer.model, "predict_proba")
    }


@app.get("/metrics")
def model_metrics():
    if scorer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": scorer.model_type,
        "scaler": scorer.scaler is not None,
        "num_encoders": len(scorer.label_encoders),
        "num_fill_values": len(scorer.fill_medians),
        "prediction_distribution": dict(PREDICTION_COUNTER),
        "total_predictions": sum(PREDICTION_COUNTER.values()),
        "training_metrics": getattr(scorer, "training_metrics", None),
        "training_distribution": getattr(scorer, "training_distribution", None),
        "current_distribution": dict(PREDICTION_COUNTER)

    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if scorer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([r.dict() for r in request.records])

    try:
        preds = scorer.predict(df).tolist()
        PREDICTION_COUNTER.update(preds)

        response = {
            "predictions": preds,
            "model_type": scorer.model_type
        }

        if hasattr(scorer.model, "predict_proba"):
            probs = scorer.predict_proba(df)
            class_names = scorer.model.classes_
            prob_list = [
                {f"Prob_{cls}": float(p[i]) for i, cls in enumerate(class_names)}
                for p in probs
            ]
            response["probabilities"] = prob_list

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
