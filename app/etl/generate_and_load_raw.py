import uuid, random
from datetime import datetime, timedelta
import numpy as np, pandas as pd
from sqlalchemy import text
from app.etl.db import get_engine

# ---------- settings ----------
N = int(50_000)   # change if you want more/less
DAYS_BACK = 30
random.seed(7); np.random.seed(7)

# ---------- generate ----------
start = datetime.now() - timedelta(days=DAYS_BACK)
accounts  = [f"A{str(i).zfill(6)}" for i in range(1, 6001)]
merchants = ["AMAZON", "UBER", "WALMART", "BESTBUY", "APPLE", "SPOTIFY",
             "AIRBNB", "STARBUCKS", "COSTCO", "SHELL", "DELTA", "NIKE"]
channels  = ["POS", "ECOM", "ATM", "P2P"]
countries = ["CA", "US", "GB", "IN", "DE", "FR", "BR", "MX"]

def sample_time():
    # uniform over 30 days
    sec = random.randint(0, DAYS_BACK*24*3600)
    return start + timedelta(seconds=sec)

def sample_amount():
    # mixture: many small, few large
    if random.random() < 0.85:
        return round(np.random.gamma(shape=2.0, scale=20.0), 2)   # mostly <$200
    else:
        return round(np.random.lognormal(mean=5.0, sigma=0.6), 2) # occasional big amounts

rows = []
for _ in range(N):
    t = sample_time()
    amt = max(0.50, sample_amount())
    acc = random.choice(accounts)
    mer = random.choice(merchants)
    ch  = random.choices(channels, weights=[0.55,0.35,0.05,0.05])[0]
    ctry= random.choices(countries, weights=[0.7,0.15,0.03,0.04,0.02,0.02,0.02,0.02])[0]

    # simple fraud logic (creates signal for the model later)
    is_night = t.hour < 6 or t.hour > 22
    high_amt = amt > 500
    risky_ctry = ctry not in ("CA","US","GB")
    risky_combo = (ch == "ECOM" and (is_night or high_amt)) or (ch=="P2P" and high_amt)

    base_prob = 0.006  # ~0.6% base rate
    prob = base_prob + (0.03 if risky_combo else 0) + (0.01 if risky_ctry else 0)
    label = np.random.rand() < min(prob, 0.25)

    rows.append({
        "tx_id": str(uuid.uuid4()),
        "tx_time": t,
        "account_id": acc,
        "merchant": mer,
        "amount": round(amt,2),
        "channel": ch,
        "country": ctry,
        "is_fraud": bool(label),
    })

df = pd.DataFrame(rows)

# ---------- load to Postgres ----------
engine = get_engine()
with engine.begin() as conn:
    # safety: ensure table exists
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS transactions_raw (
          tx_id VARCHAR PRIMARY KEY,
          tx_time TIMESTAMP NOT NULL,
          account_id VARCHAR NOT NULL,
          merchant VARCHAR NOT NULL,
          amount NUMERIC(12,2) NOT NULL,
          channel VARCHAR NOT NULL,
          country VARCHAR NOT NULL,
          is_fraud BOOLEAN NOT NULL
        )
    """))

df.to_sql("transactions_raw", con=get_engine(), if_exists="append",
          index=False, method="multi", chunksize=5000)

print(f"Inserted {len(df):,} rows into transactions_raw")
