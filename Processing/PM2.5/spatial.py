from db import get_engine
from sqlalchemy import text

def create_gold_enriched_table():
    engine = get_engine()
    
    # This query selects all columns from the silver table (*) 
    # and appends the specific spatial columns from the bronze metadata table.
    sql_query = """
    -- Ensure the gold schema exists
    CREATE SCHEMA IF NOT EXISTS gold;

    -- Drop the table if you want a fresh overwrite, or use a VIEW for real-time data
    DROP TABLE IF EXISTS gold.airqo_daily_averages_enriched;

    CREATE TABLE gold.airqo_daily_averages_enriched AS
    SELECT 
        avg_tbl.*, 
        meta.lat, 
        meta.lon, 
        meta.geom,
        meta.sub_county
    FROM silver.airqo_daily_averages_qc AS avg_tbl
    INNER JOIN bronze.airqo_metadata_sites AS meta 
        ON avg_tbl.site_id = meta.site_id;
    
    -- Optional: Add a spatial index to the new table for faster mapping
    CREATE INDEX idx_gold_geom ON gold.airqo_daily_averages_enriched USING GIST(geom);
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(sql_query))
        print("Successfully created gold.airqo_daily_averages_enriched using SQL join.")
    except Exception as e:
        print(f"Error during SQL execution: {e}")

if __name__ == "__main__":
    create_gold_enriched_table()