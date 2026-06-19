import pandas as pd
from sqlalchemy import create_engine, text
from config import db_config

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(db_config.sqlalchemy_url, pool_pre_ping=True)
    return _engine

def read_sql(query):
    engine = get_engine()
    return pd.read_sql(query, engine)

def write_table(df, table_name, schema="silver"):
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))
    
    df.to_sql(table_name, engine, schema=schema, if_exists="replace", index=False)
    print(f"Successfully updated table: {schema}.{table_name}")