import os
import psycopg2
from pathlib import Path

def get_conn():
    """Establishes a connection to the PostgreSQL database using environment variables."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "airquality"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")
    )

def ensure_bronze_tables(conn):
    """Initializes the database schema, enables PostGIS, and creates the bronze tables[cite: 2]."""
    with conn.cursor() as cur:
        # Enable PostGIS and create schema[cite: 2]
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis CASCADE;")
        cur.execute("CREATE SCHEMA IF NOT EXISTS bronze;")

    # Updated calibrated table with metadata and geometry columns[cite: 2]
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bronze.airqo_calibrated (
                datetime TIMESTAMP WITH TIME ZONE NOT NULL,
                device_name TEXT NOT NULL,
                frequency TEXT,
                humidity DOUBLE PRECISION,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                network TEXT,
                pm10 DOUBLE PRECISION,
                pm10_calibrated_value DOUBLE PRECISION,
                pm2_5 DOUBLE PRECISION,
                pm2_5_calibrated_value DOUBLE PRECISION,
                site_name TEXT,
                temperature DOUBLE PRECISION,
                site_id TEXT,
                city_name TEXT,
                source_file TEXT,
                batch_id TEXT,
                ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                geom GEOMETRY(Point, 4326),
                PRIMARY KEY (device_name, datetime)
            );
        """)

        # Raw table alignment[cite: 2]
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bronze.airqo_raw (
                datetime TIMESTAMP WITH TIME ZONE NOT NULL,
                device_name TEXT NOT NULL,
                frequency TEXT,
                humidity DOUBLE PRECISION,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                network TEXT,
                pm10 DOUBLE PRECISION,
                pm2_5 DOUBLE PRECISION,
                s1_pm10 DOUBLE PRECISION,
                s1_pm2_5 DOUBLE PRECISION,
                s2_pm10 DOUBLE PRECISION,
                s2_pm2_5 DOUBLE PRECISION,
                site_name TEXT,
                temperature DOUBLE PRECISION,
                source_file TEXT,
                batch_id TEXT,
                ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                geom GEOMETRY(Point, 4326),
                PRIMARY KEY (device_name, datetime)
            );
        """)
    conn.commit()

def _copy_into_bronze(conn, table_name, csv_path, source_file=None):
    """Internal helper to load CSV data into PostgreSQL using a temporary staging table[cite: 2]."""
    temp_table = f"stg_{table_name}"
    batch_id = os.path.basename(csv_path)

    with conn.cursor() as cur:
        # Create a temp table matching the structure of the target table[cite: 2]
        cur.execute(f"DROP TABLE IF EXISTS {temp_table};")
        cur.execute(f"CREATE TEMP TABLE {temp_table} (LIKE bronze.{table_name} INCLUDING ALL);")
        
        # 1. Drop geom from temp table so COPY works with the CSV structure (which lacks a geom column)[cite: 2]
        cur.execute(f"ALTER TABLE {temp_table} DROP COLUMN IF EXISTS geom;")

        if table_name == "airqo_calibrated":
            # Columns expected in the CSV[cite: 2]
            csv_columns = (
                "datetime, device_name, frequency, humidity, latitude, longitude, "
                "network, pm10, pm10_calibrated_value, pm2_5, pm2_5_calibrated_value, "
                "site_name, temperature, site_id, city_name"
            )
            # Full explicit list for final insertion[cite: 2]
            db_columns = f"{csv_columns}, source_file, batch_id, geom"
        else:
            csv_columns = (
                "datetime, device_name, frequency, humidity, latitude, longitude, "
                "network, pm10, pm2_5, s1_pm10, s1_pm2_5, s2_pm10, s2_pm2_5, "
                "site_name, temperature"
            )
            db_columns = f"{csv_columns}, source_file, batch_id, geom"

        # 2. Perform the COPY from the CSV into the staging table[cite: 2]
        with open(csv_path, "r", encoding="utf-8") as f:
            cur.copy_expert(
                f"COPY {temp_table} ({csv_columns}) FROM STDIN WITH (FORMAT csv, HEADER true)",
                f
            )

        # 3. Add the geom column back to the staging table to allow spatial updates[cite: 2]
        cur.execute(f"ALTER TABLE {temp_table} ADD COLUMN geom GEOMETRY(Point, 4326);")

        # 4. Update metadata and generate PostGIS geometry points[cite: 2]
        cur.execute(f"UPDATE {temp_table} SET source_file = %s, batch_id = %s", (source_file, batch_id))
        cur.execute(f"""
            UPDATE {temp_table} 
            SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
            WHERE longitude IS NOT NULL AND latitude IS NOT NULL;
        """)

        # 5. Insert into final table using explicit column lists to avoid type mismatches[cite: 2]
        cur.execute(f"""
            INSERT INTO bronze.{table_name} ({db_columns})
            SELECT {db_columns} FROM {temp_table}
            ON CONFLICT (device_name, datetime) DO NOTHING;
        """)
    conn.commit()

def load_raw_csv(conn, csv_path, source_file=None):
    """Public wrapper to load raw sensor data[cite: 2]."""
    _copy_into_bronze(conn, "airqo_raw", csv_path, source_file=source_file)

def load_calibrated_csv(conn, csv_path, source_file=None):
    """Public wrapper to load calibrated sensor data[cite: 2]."""
    _copy_into_bronze(conn, "airqo_calibrated", csv_path, source_file=source_file)