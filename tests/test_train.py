# tests/test_train.py

from app.ml.train import main as train_main
# from app.etl.features import load_training_data
from app.etl.features import load_training_data
import os

def test_training_outputs_model():
    # Run training function
    train_main()

    # Load features directly
    X, _ = load_training_data()
    assert len(X.columns) > 0

    # Check if any model files were saved
    model_files = [f for f in os.listdir("app/ml/models") if f.endswith(".joblib")]
    assert len(model_files) > 0
