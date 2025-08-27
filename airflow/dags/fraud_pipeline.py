from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "you",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Runs once a day at 02:00 UTC; no backfill by default
with DAG(
    dag_id="fraud_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 2 * * *",
    catchup=False,
    tags=["fraud", "ml"],
) as dag:

    # Train a new model (writes into app/ml/models/)
    train_model = BashOperator(
        task_id="train_model",
        bash_command="python -m app.ml.train",
        env={},  # inherits container env (your DB_* from .env)
    )

    # Batch-score any unscored txs and write to predictions table
    batch_score = BashOperator(
        task_id="batch_score",
        bash_command="python -m app.ml.predict_batch",
        env={},
    )

    train_model >> batch_score
