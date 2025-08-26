# tests/test_api.py
from fastapi.testclient import TestClient
import app.api.main as main
import numpy as np

# ---- tiny stubs so we don't need a real DB or real model ----

class _NullConn:
    def execute(self, *args, **kwargs):
        # swallow INSERT into predictions during tests
        return None
    def __enter__(self): return self
    def __exit__(self, *exc): pass

class _DummyEngine:
    def begin(self):
        # mimic SQLAlchemy engine.begin() as context manager
        return _NullConn()

class _DummyModel:
    def predict_proba(self, X):
        # return fixed probability for the positive class
        # shape: (n_rows, 2) -> [p(class0), p(class1)]
        p = 0.6
        return np.column_stack([1 - p, np.full(len(X), p)])

def test_prediction():
    # 1) Bypass startup model loading by directly setting globals
    main.MODEL = _DummyModel()
    main.FEATURES = main.X_COLS

    # 2) Bypass DB insert in /predict
    main.get_engine = lambda: _DummyEngine()

    client = TestClient(main.app)

    sample = {
        "tx_id": "test123",
        "amount": 100.0,
        "hour": 14,
        "is_night": False,
        "channel_onehot": "web",
        "country_onehot": "US",
        "rolling_amt_1h": 300.5,
        "rolling_cnt_1h": 2
    }
    resp = client.post("/predict", json=sample)
    assert resp.status_code == 200
    body = resp.json()
    assert body["tx_id"] == "test123"
    assert 0 <= body["prob_fraud"] <= 1
