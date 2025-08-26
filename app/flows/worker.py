# app/flows/worker.py
import subprocess, time, datetime as dt

SCORE_EVERY_SECS = 60      # run predict_batch once a minute
RETRAIN_EVERY_SECS = 24*3600  # optional: daily retrain

def run(cmd):
    print(f"[{dt.datetime.utcnow().isoformat()}Z] -> {cmd}")
    subprocess.run(cmd, check=False)

def main():
    last_retrain = 0
    while True:
        # 1) Score any unscored rows
        run(["python", "-m", "app.ml.predict_batch"])

        # 2) Optional: retrain once per day
        now = time.time()
        if now - last_retrain > RETRAIN_EVERY_SECS:
            # comment out if you don't want auto-retrain yet
            run(["python", "-m", "app.ml.train"])
            last_retrain = now

        time.sleep(SCORE_EVERY_SECS)

if __name__ == "__main__":
    main()
