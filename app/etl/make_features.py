# app/etl/make_features.py
import pandas as pd
from sqlalchemy import text
from app.etl.db import get_engine

def main():
    eng = get_engine()

    # 1) Load raw data
    q = """
      SELECT tx_id, tx_time, account_id, merchant, amount, channel, country, is_fraud
      FROM transactions_raw
      ORDER BY account_id, tx_time
    """
    df = pd.read_sql(q, eng, parse_dates=["tx_time"])

    # 2) Stable order + simple index
    df = df.sort_values(["account_id", "tx_time"]).reset_index(drop=True)

    # 3) Rolling features per account over last 1 hour
    def add_rollings(g):
        g = g.sort_values("tx_time")
        r_amt = g.rolling("1h", on="tx_time")["amount"].sum()
        r_cnt = g.rolling("1h", on="tx_time")["amount"].count()
        g["rolling_amt_1h"] = r_amt
        g["rolling_cnt_1h"] = r_cnt
        return g

    df = df.groupby("account_id", group_keys=False).apply(add_rollings)

    # 4) Supervised label
    df["label"] = df["is_fraud"].astype(bool)

    # 5) Columns your table expects (and only those)
    df["hour"] = df["tx_time"].dt.hour
    df["is_night"] = (df["hour"] < 6) | (df["hour"] > 22)
    df["channel_onehot"] = df["channel"]     # placeholder encoding
    df["country_onehot"] = df["country"]     # placeholder encoding

    feats = df[[
        "tx_id", "tx_time", "amount",
        "hour", "is_night",
        "channel_onehot", "country_onehot",
        "rolling_amt_1h", "rolling_cnt_1h",
        "label"
    ]]

    # 6) Load (truncate then append)
    with eng.begin() as con:
        con.execute(text("TRUNCATE TABLE transactions_features"))
        feats.to_sql(
            "transactions_features",
            con=con,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000,
        )

    print(f"Features saved: {len(feats):,}")

if __name__ == "__main__":
    main()
