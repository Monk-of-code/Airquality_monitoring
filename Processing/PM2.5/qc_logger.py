from sqlalchemy import text
from db import get_engine
from config import db_config

def init_qc_log():
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS silver;"))
        conn.execute(text(f"DROP TABLE IF EXISTS silver.{db_config.qc_log_table};"))
        conn.execute(text(f"""
            CREATE TABLE silver.{db_config.qc_log_table} (
                id SERIAL PRIMARY KEY,
                check_name VARCHAR(255),
                rows_before INT,
                rows_passed INT,
                rows_flagged INT,
                logged_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))

def log_qc(check_name, rows_before, rows_passed, rows_flagged):
    engine = get_engine()
    sql = f"INSERT INTO silver.{db_config.qc_log_table} (check_name, rows_before, rows_passed, rows_flagged) VALUES (:n, :b, :p, :f)"
    with engine.begin() as conn:
        conn.execute(text(sql), {"n": check_name, "b": rows_before, "p": rows_passed, "f": rows_flagged})