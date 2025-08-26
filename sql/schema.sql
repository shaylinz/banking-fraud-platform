CREATE TABLE IF NOT EXISTS transactions_raw (
  tx_id VARCHAR PRIMARY KEY,
  tx_time TIMESTAMP NOT NULL,
  account_id VARCHAR NOT NULL,
  merchant VARCHAR NOT NULL,
  amount NUMERIC(12,2) NOT NULL,
  channel VARCHAR NOT NULL,
  country VARCHAR NOT NULL,
  is_fraud BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions_features (
  tx_id VARCHAR PRIMARY KEY,
  tx_time TIMESTAMP NOT NULL,
  amount NUMERIC(12,2),
  hour INT,
  is_night BOOLEAN,
  channel_onehot JSONB,
  country_onehot JSONB,
  rolling_amt_1h NUMERIC(12,2),
  rolling_cnt_1h INT,
  label BOOLEAN
);

CREATE TABLE IF NOT EXISTS predictions (
  tx_id VARCHAR PRIMARY KEY,
  scored_at TIMESTAMP NOT NULL DEFAULT NOW(),
  prob_fraud NUMERIC(6,5),
  predicted BOOLEAN
);
