# app/etl/db.py
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

def get_engine():
    """
    Create and return an SQLAlchemy engine using values from .env
    """
    load_dotenv()  # read .env from project root
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    user = os.getenv("DB_USER", "postgres")
    pwd  = os.getenv("DB_PASSWORD", "postgres")
    db   = os.getenv("DB_NAME", "frauddb")

    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
    return create_engine(url, future=True)
