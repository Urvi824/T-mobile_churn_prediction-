from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import joblib, json, numpy as np

app = FastAPI(title="T‑Mobile Churn API", version="1.0.0")

PIPELINE_PATH = "churn_pipeline.joblib"
INFO_PATH = "model_info.json"

try:
    model = joblib.load(PIPELINE_PATH)
    model_info = json.load(open(INFO_PATH))
except Exception as e:
    model, model_info = None, None
    print("Load error:", e)

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]

class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: Optional[List[float]] = None

@app.get("/health")
def health():
    return {"ok": model is not None, "sklearn": model_info.get("sklearn_version") if model_info else None}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    X = req.records
    probs = model.predict_proba(X)[:,1].tolist()
    preds = (np.array(probs) >= 0.5).astype(int).tolist()
    return PredictResponse(predictions=preds, probabilities=probs)

@app.get("/info")
def info():
    return model_info or {}
