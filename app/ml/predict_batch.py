# app/ml/predict_batch.py
import datetime as dt
import joblib
import pandas as pd
from sqlalchemy import text
from app.etl.db import get_engine

# These are the *intended* features you pull from the DB for scoring.
# We'll still re-order them to exactly match the model's training order.
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
    return con.execute(q).scalar_one()

def fetch_batch(con, limit=10000):
    # Only score rows that are not in predictions yet
    q = text(f"""
        SELECT tf.tx_id, tf.tx_time, tf.{', tf.'.join(X_COLS)}
        FROM transactions_features tf
        LEFT JOIN predictions p ON p.tx_id = tf.tx_id
        WHERE p.tx_id IS NULL
        ORDER BY tf.tx_time
        LIMIT :limit
    """)
    return pd.read_sql(q, con, params={"limit": limit})

def resolve_feature_order(bundle, model, cols_available):
    """
    Work out the exact feature order the trained pipeline expects.
    Prefer bundle['feature_order']; otherwise use model.feature_names_in_
    if present. Validate vs the columns we fetched from DB.
    """
    feature_order = None

    # Preferred: we saved it during training
    if isinstance(bundle, dict):
        feature_order = bundle.get("feature_order")

    # Fallback: some sklearn estimators expose the learned input names
    if feature_order is None and hasattr(model, "feature_names_in_"):
        feature_order = list(model.feature_names_in_)

    if feature_order is None:
        raise RuntimeError(
            "Cannot determine feature order. "
            "Save it in train.py as bundle['feature_order'] or rely on "
            "model.feature_names_in_ if available."
        )

    # Validate presence and order
    missing = [c for c in feature_order if c not in cols_available]
    if missing:
        raise ValueError(f"Missing columns for prediction: {missing}")

    return feature_order

def main():
    eng = get_engine()

    # 1) Find the latest trained model
    with eng.begin() as con:
        model_path = get_latest_model_path(con)
    print(f"Using model: {model_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]  # extract the sklearn Pipeline/estimator

    total = 0
    while True:
        # 2) Pull a batch of unscored feature rows
        with eng.begin() as con:
            batch = fetch_batch(con, limit=10000)

        if batch.empty:
            print(f"No more rows to score. Total saved: {total}")
            break

        # 3) Build feature matrix in the EXACT order the model expects
        cols_available = [c for c in batch.columns if c in X_COLS]
        feature_order = resolve_feature_order(bundle, model, cols_available)
        X = batch[feature_order].copy()

        # 4) Predict fraud probability
        proba = model.predict_proba(X)[:, 1]

        out = pd.DataFrame({
            "tx_id": batch["tx_id"],
            "scored_at": dt.datetime.utcnow(),
            "prob_fraud": proba.round(5),     # fits NUMERIC(6,5)
            "predicted": (proba >= 0.5)       # simple 0.5 threshold
        })

        # 5) Save to predictions table
        with eng.begin() as con:
            out.to_sql(
                "predictions",
                con=con,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=5000,
            )

        total += len(out)
        print(f"Scored & saved: {len(out)} (running total={total})")

if __name__ == "__main__":
    main()
