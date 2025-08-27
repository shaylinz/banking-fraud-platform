# 🏦 Banking Fraud Detection Platform

End-to-end fraud detection platform with **Airflow**, **FastAPI**, **Postgres**, and **Docker**.  

---

## 🚀 Quick Start (VS Code Terminal)

### 1️⃣ Clone the repo
```bash
git clone https://github.com/<your-username>/banking-fraud-platform.git
cd banking-fraud-platform
2️⃣ Create .env file
Create a .env file in the root folder with:

POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow



3️⃣ Build and start everything

docker compose up -d --build


4️⃣ Check running containers

docker compose ps

You should see services: postgres, pgadmin, airflow-webserver, airflow-scheduler, api, worker.

🌐 Access the services
Airflow UI → http://localhost:8080
Login: admin / admin

PgAdmin UI → http://localhost:5050
Login: admin@admin.com / admin

Fraud API (Swagger UI) → http://localhost:8000/docs

▶️ Run the DAGs
Go to http://localhost:8080 (Airflow).

Find the DAGs:

fraud_pipeline

hello_dag

Click the toggle switch to unpause.

Hit Play (Trigger DAG) to run.

Check logs/graph view for progress.

📊 Database
Open PgAdmin → http://localhost:5050.

Login with admin@admin.com / admin.

Connect to the airflow database.

Run queries to see pipeline results.

🛑 Stop everything

docker compose down -v

This will stop and remove all containers, networks, and volumes.