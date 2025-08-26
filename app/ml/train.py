# app/ml/train.py
import os, json, datetime as dt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sqlalchemy import text
from joblib import dump

from app.etl.db import get_engine

MODEL_DIR = "app/ml/models"
MODEL_NAME = "fraud_logreg"
MODEL_VERSION = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")

# --- feature layout we will use everywhere (train + predict) ---
NUM_COLS = ["amount", "hour", "is_night", "rolling_amt_1h", "rolling_cnt_1h"]
CAT_COLS = ["channel_onehot", "country_onehot"]
FEATURES = NUM_COLS + CAT_COLS
TARGET_COL = "label"


def load_features():
    eng = get_engine()
    q = """
      SELECT amount, hour, is_night, rolling_amt_1h, rolling_cnt_1h,
             channel_onehot, country_onehot, label, tx_time
      FROM transactions_features
      ORDER BY tx_time
    """
    df = pd.read_sql(q, eng, parse_dates=["tx_time"])
    return df


def preprocess(df: pd.DataFrame):
    """Select columns, basic NA handling, and return X, y in the expected order."""
    df = df.dropna(subset=FEATURES + [TARGET_COL]).copy()
    # ensure numeric dtypes for numeric columns (safety)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=NUM_COLS + [TARGET_COL])

    # categoricals stay as strings; the pipeline will one-hot encode them
    X = df[FEATURES]
    y = df[TARGET_COL].astype(int)
    return X, y


def ensure_registry_table(eng):
    with eng.begin() as con:
        con.execute(text("""
          CREATE TABLE IF NOT EXISTS model_registry (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR NOT NULL,
            version VARCHAR NOT NULL,
            trained_at TIMESTAMP NOT NULL,
            roc_auc NUMERIC(6,4),
            pr_auc NUMERIC(6,4),
            artifact_path TEXT
          );
        """))


def save_registry(eng, metrics, artifact_path):
    with eng.begin() as con:
        con.execute(
            text("""
              INSERT INTO model_registry
                (model_name, version, trained_at, roc_auc, pr_auc, artifact_path)
              VALUES
                (:model_name, :version, NOW(), :roc_auc, :pr_auc, :artifact_path)
            """),
            dict(
                model_name=MODEL_NAME,
                version=MODEL_VERSION,
                roc_auc=float(metrics["roc_auc"]),
                pr_auc=float(metrics["pr_auc"]),
                artifact_path=artifact_path,
            )
        )


def main():
    print("Loading features…")
    df = load_features()
    X, y = preprocess(df)

    # stratified random split (simple baseline)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- preprocessing + model pipeline ---
    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ],
        remainder="drop",
        sparse_threshold=1.0,  # allow sparse if needed
    )

    pipe = Pipeline([
        ("prep", preproc),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    print("Training model…")
    pipe.fit(X_train, y_train)

    # Evaluate
    proba = pipe.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)
    print(f"ROC AUC: {roc:.4f}   PR AUC: {pr:.4f}")

    # Save artifact bundle
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_{MODEL_VERSION}.joblib")
    dump(
        {
            "model": pipe,                # full pipeline (prep + clf)
            "feature_order": FEATURES,    # exact order expected at predict time
            "saved_at": dt.datetime.utcnow().isoformat() + "Z",
            "roc_auc": float(roc),
            "pr_auc": float(pr),
        },
        artifact_path,
    )
    print(f"Saved model → {artifact_path}")

    # Registry row
    eng = get_engine()
    ensure_registry_table(eng)
    save_registry(eng, {"roc_auc": roc, "pr_auc": pr}, artifact_path)

    # Optional sidecar JSON
    meta = {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "features": FEATURES,
        "metrics": {"roc_auc": roc, "pr_auc": pr},
        "saved_at": dt.datetime.utcnow().isoformat() + "Z",
        "artifact_path": artifact_path,
    }
    with open(os.path.join(MODEL_DIR, f"{MODEL_NAME}_{MODEL_VERSION}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    main()
