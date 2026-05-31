from __future__ import annotations
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException
from cardiotwin.api.schemas import ECGUpload
from cardiotwin.models.baseline_ml import load_model
from cardiotwin.pipeline.inference import run_inference

app = FastAPI(title="CardioTwin-AI 12L API", version="0.1.0")
MODEL_BUNDLE = None
MODEL_PATH = Path("artifacts/models/baseline_model.joblib")

@app.on_event("startup")
def startup():
    global MODEL_BUNDLE
    if MODEL_PATH.exists():
        MODEL_BUNDLE = load_model(MODEL_PATH)

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": MODEL_BUNDLE is not None}

@app.post("/predict")
def predict(payload: ECGUpload):
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model not found. Train baseline and save artifacts/models/baseline_model.joblib first.")
    x = np.asarray(payload.samples, dtype=np.float32)
    if x.ndim != 2:
        raise HTTPException(status_code=400, detail="samples must be 2D: rows=samples, cols=leads")
    state = run_inference(MODEL_BUNDLE, x, payload.fs, payload.leads, record_id=payload.record_id)
    return {"record_id": payload.record_id, "state": state}
