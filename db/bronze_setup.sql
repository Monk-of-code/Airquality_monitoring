CREATE EXTENSION IF NOT EXISTS postgis CASCADE; 
CREATE SCHEMA IF NOT EXISTS bronze;


-- =========================
-- bronze.airqo_calibrated
-- =========================
CREATE TABLE IF NOT EXISTS bronze.airqo_calibrated (
    datetime TIMESTAMP WITH TIME ZONE NOT NULL,
    device_name TEXT,
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

  -- Auto-updating geometry column
  geom GEOMETRY(Point, 4326) GENERATED ALWAYS AS (
    CASE 
      WHEN longitude IS NOT NULL AND latitude IS NOT NULL 
      THEN ST_SetSRID(ST_MakePoint(longitude, latitude), 4326) 
      ELSE NULL 
    END
  ) STORED,

  source_file   TEXT,
  batch_id      TEXT,
  ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT airqo_cal_pk PRIMARY KEY (device_name, datetime)
);



CREATE INDEX IF NOT EXISTS airqo_cal_geom_idx ON bronze.airqo_calibrated USING GIST (geom);
CREATE INDEX IF NOT EXISTS airqo_cal_device_ts_idx ON bronze.airqo_calibrated (device_name, datetime DESC);

