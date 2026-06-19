from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from psycopg2.extras import Json


def build_sqlalchemy_url_from_env() -> str:
    try:
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        user = os.getenv("POSTGRES_USER", "airqouser")
        password = os.getenv("POSTGRES_PASSWORD", "airqopassword")
        db = os.getenv("POSTGRES_DB", "airqodb")

        if not all([host, port, user, password, db]):
            raise ValueError("Missing one or more required environment variables for Postgres connection.")

        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

    except Exception as e:
        raise RuntimeError(f"build_sqlalchemy_url_from_env failed: {e}") from e


@dataclass(frozen=True)
class PostgresConfig:
    sqlalchemy_url: str
    schema: str = "bronze"
    grids_table: str = "airqo_metadata_grids"
    sites_table: str = "airqo_metadata_sites"
    site_devices_table: str = "airqo_metadata_site_devices"


def get_engine(cfg: PostgresConfig) -> Engine:
    try:
        return create_engine(cfg.sqlalchemy_url, pool_pre_ping=True)
    except Exception as e:
        raise RuntimeError(f"get_engine failed: {e}") from e


def ensure_base(engine: Engine, cfg: PostgresConfig) -> None:
    """
    Ensures required extensions + schema exist.
    """
    try:
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS postgis CASCADE;
        CREATE SCHEMA IF NOT EXISTS {cfg.schema};
        """
        with engine.begin() as conn:
            conn.execute(text(ddl))
    except Exception as e:
        raise RuntimeError(f"ensure_base failed: {e}") from e


# ----------------------------
# Grids
# ----------------------------
def ensure_grids_table(engine: Engine, cfg: PostgresConfig) -> None:
    try:
        ensure_base(engine, cfg)

        ddl = f"""
        CREATE TABLE IF NOT EXISTS {cfg.schema}.{cfg.grids_table} (
            grid_id TEXT PRIMARY KEY,
            name TEXT,
            admin_level TEXT,
            visibility BOOLEAN,
            network TEXT,
            long_name TEXT,
            created_at TIMESTAMPTZ,
            payload JSONB,
            ingested_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        with engine.begin() as conn:
            conn.execute(text(ddl))
    except Exception as e:
        raise RuntimeError(f"ensure_grids_table failed: {e}") from e


def upsert_grids(
    grids_df: pd.DataFrame,
    cfg: PostgresConfig,
    engine: Optional[Engine] = None,
) -> int:
    try:
        if grids_df is None or grids_df.empty:
            return 0

        required = {"grid_id", "payload"}
        missing = required - set(grids_df.columns)
        if missing:
            raise ValueError(f"grids_df missing columns: {sorted(list(missing))}")

        _engine = engine or get_engine(cfg)
        ensure_grids_table(_engine, cfg)

        sql = text(f"""
            INSERT INTO {cfg.schema}.{cfg.grids_table}
                (grid_id, name, admin_level, visibility, network, long_name, created_at, payload, ingested_at)
            VALUES
                (:grid_id, :name, :admin_level, :visibility, :network, :long_name, :created_at, :payload, now())
            ON CONFLICT (grid_id)
            DO UPDATE SET
                name        = EXCLUDED.name,
                admin_level = EXCLUDED.admin_level,
                visibility  = EXCLUDED.visibility,
                network     = EXCLUDED.network,
                long_name   = EXCLUDED.long_name,
                created_at  = EXCLUDED.created_at,
                payload     = EXCLUDED.payload,
                ingested_at = now();
        """)

        rows = []
        for r in grids_df.to_dict(orient="records"):
            r["payload"] = Json(r["payload"])
            rows.append(r)

        with _engine.begin() as conn:
            conn.execute(sql, rows)

        return int(len(rows))

    except Exception as e:
        raise RuntimeError(f"upsert_grids failed: {e}") from e


# ----------------------------
# Sites (WITH geom)
# ----------------------------
def ensure_sites_table(engine: Engine, cfg: PostgresConfig) -> None:
    try:
        ensure_base(engine, cfg)

        ddl = f"""
        CREATE TABLE IF NOT EXISTS {cfg.schema}.{cfg.sites_table} (
            site_id TEXT PRIMARY KEY,
            grid_id TEXT,
            name TEXT,
            formatted_name TEXT,
            country TEXT,
            region TEXT,
            city TEXT,
            county TEXT,
            sub_county TEXT,
            parish TEXT,
            lat DOUBLE PRECISION,
            lon DOUBLE PRECISION,

            -- Generated geometry (same approach as your bronze tables)
            geom GEOMETRY(Point, 4326) GENERATED ALWAYS AS (
              CASE
                WHEN lon IS NOT NULL AND lat IS NOT NULL
                THEN ST_SetSRID(ST_MakePoint(lon, lat), 4326)
                ELSE NULL
              END
            ) STORED,

            location_name TEXT,
            search_name TEXT,
            data_provider TEXT,
            payload JSONB,
            ingested_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS {cfg.sites_table}_geom_idx
          ON {cfg.schema}.{cfg.sites_table} USING GIST (geom);

        CREATE INDEX IF NOT EXISTS {cfg.sites_table}_grid_id_idx
          ON {cfg.schema}.{cfg.sites_table} (grid_id);
        """
        with engine.begin() as conn:
            conn.execute(text(ddl))
    except Exception as e:
        raise RuntimeError(f"ensure_sites_table failed: {e}") from e


def upsert_sites(
    sites_df: pd.DataFrame,
    cfg: PostgresConfig,
    engine: Optional[Engine] = None,
) -> int:
    try:
        if sites_df is None or sites_df.empty:
            return 0

        required = {"site_id", "payload"}
        missing = required - set(sites_df.columns)
        if missing:
            raise ValueError(f"sites_df missing columns: {sorted(list(missing))}")

        _engine = engine or get_engine(cfg)
        ensure_sites_table(_engine, cfg)

        sql = text(f"""
            INSERT INTO {cfg.schema}.{cfg.sites_table}
                (site_id, grid_id, name, formatted_name, country, region, city, county, sub_county, parish,
                 lat, lon, location_name, search_name, data_provider, payload, ingested_at)
            VALUES
                (:site_id, :grid_id, :name, :formatted_name, :country, :region, :city, :county, :sub_county, :parish,
                 :lat, :lon, :location_name, :search_name, :data_provider, :payload, now())
            ON CONFLICT (site_id)
            DO UPDATE SET
                grid_id        = EXCLUDED.grid_id,
                name           = EXCLUDED.name,
                formatted_name = EXCLUDED.formatted_name,
                country        = EXCLUDED.country,
                region         = EXCLUDED.region,
                city           = EXCLUDED.city,
                county         = EXCLUDED.county,
                sub_county     = EXCLUDED.sub_county,
                parish         = EXCLUDED.parish,
                lat            = EXCLUDED.lat,
                lon            = EXCLUDED.lon,
                location_name  = EXCLUDED.location_name,
                search_name    = EXCLUDED.search_name,
                data_provider  = EXCLUDED.data_provider,
                payload        = EXCLUDED.payload,
                ingested_at    = now();
        """)

        rows = []
        for r in sites_df.to_dict(orient="records"):
            r["payload"] = Json(r["payload"])
            rows.append(r)

        with _engine.begin() as conn:
            conn.execute(sql, rows)

        return int(len(rows))

    except Exception as e:
        raise RuntimeError(f"upsert_sites failed: {e}") from e


# ----------------------------
# Site↔Device mapping
# ----------------------------
def ensure_site_devices_table(engine: Engine, cfg: PostgresConfig) -> None:
    """
    Stores mapping facts discovered from generate endpoint.
    """
    try:
        ensure_base(engine, cfg)

        ddl = f"""
        CREATE TABLE IF NOT EXISTS {cfg.schema}.{cfg.site_devices_table} (
            grid_id TEXT NOT NULL,
            site_id TEXT NULL,
            device_id TEXT NULL,
            payload JSONB,
            ingested_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (grid_id, site_id, device_id)
        );
        """
        with engine.begin() as conn:
            conn.execute(text(ddl))
    except Exception as e:
        raise RuntimeError(f"ensure_site_devices_table failed: {e}") from e


def upsert_site_devices(
    site_devices_df: pd.DataFrame,
    cfg: PostgresConfig,
    engine: Optional[Engine] = None,
) -> int:
    try:
        if site_devices_df is None or site_devices_df.empty:
            return 0

        required = {"grid_id", "site_id", "device_id", "payload"}
        missing = required - set(site_devices_df.columns)
        if missing:
            raise ValueError(f"site_devices_df missing columns: {sorted(list(missing))}")

        _engine = engine or get_engine(cfg)
        ensure_site_devices_table(_engine, cfg)

        sql = text(f"""
            INSERT INTO {cfg.schema}.{cfg.site_devices_table}
                (grid_id, site_id, device_id, payload, ingested_at)
            VALUES
                (:grid_id, :site_id, :device_id, :payload, now())
            ON CONFLICT (grid_id, site_id, device_id)
            DO UPDATE SET
                payload     = EXCLUDED.payload,
                ingested_at = now();
        """)

        rows = []
        for r in site_devices_df.to_dict(orient="records"):
            r["payload"] = Json(r["payload"])
            rows.append(r)

        with _engine.begin() as conn:
            conn.execute(sql, rows)

        return int(len(rows))

    except Exception as e:
        raise RuntimeError(f"upsert_site_devices failed: {e}") from e