from __future__ import annotations
from pathlib import Path
from typing import Literal, Tuple
import pandas as pd
import csv

Level = Literal["raw", "calibrated"]

#  column ordering for database alignment
RAW_COLS = [
    "datetime", "device_name", "frequency", "humidity", "latitude", "longitude",
    "network", "pm10", "pm2_5", "s1_pm10", "s1_pm2_5", "s2_pm10", "s2_pm2_5",
    "site_name", "temperature"
]

CAL_COLS = [
    "datetime", "device_name", "frequency", "humidity", "latitude", "longitude",
    "network", "pm10", "pm10_calibrated_value", "pm2_5", "pm2_5_calibrated_value",
    "site_name", "temperature", "site_id", "city_name"
]

#  Detect delimiter automatically (CSV vs TSV)
def detect_delimiter(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        sample = f.read(2048)
        try:
            dialect = csv.Sniffer().sniff(sample)
            return dialect.delimiter
        except Exception:
            return "\t"  # fallback (your data is TSV)


def clean_and_write(src: Path, out_root: Path) -> Tuple[Path, Level, int]:
    out_root.mkdir(parents=True, exist_ok=True)

    #  Detect delimiter
    delimiter = detect_delimiter(src)

    #  Read correctly
    df = pd.read_csv(src, sep=delimiter)

    #  Safety check
    if len(df.columns) == 1:
        raise ValueError(f"{src.name} not parsed correctly — delimiter issue")

    # Detect level
    level: Level = "calibrated" if "pm10_calibrated_value" in df.columns else "raw"
    # Ensure we use the new CAL_COLS list
    target_cols = RAW_COLS if level == "raw" else CAL_COLS

    # Timezone normalization: Keeping as TIMESTAMPTZ for the new SQL schema
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["datetime"] = df["datetime"].dt.tz_convert("Africa/Nairobi")
    # Numeric cleaning
    numeric_cols = [
        "humidity", "latitude", "longitude", "pm10", "pm2_5",
        "temperature", "pm10_calibrated_value", "pm2_5_calibrated_value"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix fake coordinates
    # 1. First, ensure columns are numeric so the comparison works[cite: 1]
    # Fix fake coordinates

# 1. Ensure columns are numeric
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("'", "").str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")

# 2. Apply "Null Island" fix
    if "latitude" in df.columns and "longitude" in df.columns:
        mask = (df["latitude"] == 0) & (df["longitude"] == 0)
        df.loc[mask, ["latitude", "longitude"]] = pd.NA

# Schema alignment: Ensure site_id and city_name exist
    for col in target_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Select + deduplicate
    df = df[target_cols].drop_duplicates(subset=["datetime", "device_name"])

    out_path = out_root / f"cleaned_{src.name}"
    df.to_csv(out_path, index=False)

    return out_path, level, len(df)