import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Path to your .env file
ENV_PATH = Path("/home/wanji/projects/airquality/.env")
load_dotenv(dotenv_path=ENV_PATH)

def build_sqlalchemy_url_from_env() -> str:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD")
    db = os.getenv("POSTGRES_DB", "airquality")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

@dataclass(frozen=True)
class PostgresConfig:
    sqlalchemy_url: str = build_sqlalchemy_url_from_env()
    schema: str = "bronze"
    bronze_cal_table: str = "airqo_calibrated"
    
    # Silver Table Names
    daily_avg_table: str = "airqo_daily_averages_qc"
    threshold_rpt_table: str = "airqo_threshold_sensitivity_report"
    annual_rpt_table: str = "airqo_annual_75pct_report"
    qc_log_table: str = "airqo_qc_log"

db_config = PostgresConfig()