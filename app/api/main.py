# app/api/main.py
import datetime as dt
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.etl.db import get_engine  # reuses your DB helper

# --- NEW: repo root inside the container ---
APP_ROOT = Path("/app")

# Must match training order
X_COLS = [
    "amount",
    "hour",
    "is_night",
    "channel_onehot",
    "country_onehot",
    "rolling_amt_1h",
    "rolling_cnt_1h",
]

def get_latest_model_path(con):
    q = text("""
        SELECT artifact_path
        FROM model_registry
        ORDER BY trained_at DESC
        LIMIT 1
    """)
    p = con.execute(q).scalar_one()

    # Normalize Windows path to Linux style and make it absolute under /app
    p = p.replace("\\", "/")
    if not p.startswith("/"):
        p = str((APP_ROOT / p).resolve())
    return p

app = FastAPI(title="Fraud API")

MODEL = None
FEATURES = X_COLS

@app.on_event("startup")
def _load_model():
    """Load the newest model once when the API starts."""
    global MODEL, FEATURES
    eng = get_engine()
    with eng.begin() as con:
        path = get_latest_model_path(con)
    bundle = joblib.load(path)           # dict(model=..., features=[...])
    MODEL = bundle["model"]
    FEATURES = bundle.get("features", X_COLS)
    print(f"Loaded model from {path} with features={FEATURES}")

class TxIn(BaseModel):
    tx_id: str
    amount: float
    hour: int
    is_night: bool
    channel_onehot: str
    country_onehot: str
    rolling_amt_1h: float
    rolling_cnt_1h: float
    tx_time: dt.datetime = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc)
    )

class PredictOut(BaseModel):
    tx_id: str
    prob_fraud: float
    predicted: bool

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict", response_model=PredictOut)
def predict(payload: TxIn):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Keep feature order identical to training
    X = pd.DataFrame([{c: getattr(payload, c) for c in FEATURES}])

    # Predict
    proba = float(MODEL.predict_proba(X)[:, 1][0])
    pred = proba >= 0.5

    # Save to DB (upsert on tx_id so re-calls just update)
    eng = get_engine()
    with eng.begin() as con:
        con.execute(text("""
            INSERT INTO predictions (tx_id, scored_at, prob_fraud, predicted)
            VALUES (:tx_id, NOW(), :prob, :pred)
            ON CONFLICT (tx_id) DO UPDATE
              SET scored_at = EXCLUDED.scored_at,
                  prob_fraud = EXCLUDED.prob_fraud,
                  predicted  = EXCLUDED.predicted
        """), {"tx_id": payload.tx_id, "prob": round(proba, 5), "pred": pred})

    return PredictOut(tx_id=payload.tx_id, prob_fraud=round(proba, 5), predicted=pred)
