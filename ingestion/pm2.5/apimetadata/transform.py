from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd


# ----------------------------
# Grids
# ----------------------------
def grids_to_df(raw: Dict[str, Any]) -> pd.DataFrame:
    """
    raw expected: { success: bool, grids: [ ... ] }
    Output: one row per grid_id.
    """
    try:
        grids = raw.get("grids") or []
        rows: List[Dict[str, Any]] = []

        for g in grids:
            grid_id = g.get("_id") or g.get("id")
            rows.append(
                {
                    "grid_id": None if grid_id is None else str(grid_id),
                    "name": g.get("name"),
                    "admin_level": g.get("admin_level"),
                    "visibility": g.get("visibility"),
                    "network": g.get("network"),
                    "long_name": g.get("long_name") or g.get("longName"),
                    "created_at": g.get("createdAt") or g.get("created_at") or g.get("created_at_utc"),
                    "payload": g,
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
            df = (
                df.dropna(subset=["grid_id"])
                  .drop_duplicates(subset=["grid_id"])
                  .reset_index(drop=True)
            )

        return df

    except Exception as e:
        raise RuntimeError(f"grids_to_df failed: {e}") from e


# ----------------------------
# Sites (from grids/summary)
# ----------------------------
def grids_summary_to_sites_df(raw: Dict[str, Any]) -> pd.DataFrame:
    """
    raw expected: { success: bool, grids: [ { _id, sites: [...] } ] }

    Output columns:
      site_id (PK), grid_id, name, generated_name, formatted_name,
      country, region, city, county, sub_county, parish, lat, lon,
      location_name, search_name, data_provider, payload
    """
    try:
        grids = raw.get("grids") or []
        rows: List[Dict[str, Any]] = []

        for g in grids:
            grid_id = g.get("_id") or g.get("id")
            sites = g.get("sites") or []

            for s in sites:
                site_id = s.get("_id") or s.get("id")
                rows.append(
                    {
                        "grid_id": None if grid_id is None else str(grid_id),
                        "site_id": None if site_id is None else str(site_id),
                        "name": s.get("name"),
                        "generated_name": s.get("generated_name"),
                        "formatted_name": s.get("formatted_name"),
                        "country": s.get("country"),
                        "region": s.get("region"),
                        "city": s.get("city"),
                        "county": s.get("county"),
                        "sub_county": s.get("sub_county"),
                        "parish": s.get("parish"),
                        "lat": s.get("approximate_latitude"),
                        "lon": s.get("approximate_longitude"),
                        "location_name": s.get("location_name"),
                        "search_name": s.get("search_name"),
                        "data_provider": s.get("data_provider"),
                        "payload": s,
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            df = (
                df.dropna(subset=["site_id"])
                  .drop_duplicates(subset=["site_id"])
                  .reset_index(drop=True)
            )

        return df

    except Exception as e:
        raise RuntimeError(f"grids_summary_to_sites_df failed: {e}") from e


# ----------------------------
# Site↔Device mapping (from grids/{id}/generate)
# ----------------------------
def grid_generate_to_site_devices_df(raw: Dict[str, Any], grid_id: str) -> pd.DataFrame:
    """
    Handles generate response shapes including:
      - { sites_and_devices: { site_ids: [...], device_ids: [...] } }
      - { data: { site_ids: [...], device_ids: [...] } }
      - { site_ids: [...], device_ids: [...] }
      - list of explicit pairs (rare)

    Output: [grid_id, site_id, device_id, payload]
    Note: If no explicit pairing is provided, we store:
      - rows for sites (device_id NULL)
      - rows for devices (site_id NULL)
    """
    try:
        grid_id_str = str(grid_id)

        # ---- NEW: AirQo current shape
        if isinstance(raw.get("sites_and_devices"), dict):
            data = raw["sites_and_devices"]
        else:
            data = raw.get("data", raw)

        # explicit list mapping case
        if isinstance(data, list):
            rows = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                sid = item.get("site_id") or item.get("siteId") or item.get("site")
                did = item.get("device_id") or item.get("deviceId") or item.get("device")
                if sid is None and did is None:
                    continue
                rows.append(
                    {
                        "grid_id": grid_id_str,
                        "site_id": None if sid is None else str(sid),
                        "device_id": None if did is None else str(did),
                        "payload": item,
                    }
                )
            df = pd.DataFrame(rows)
            if df.empty:
                return pd.DataFrame(columns=["grid_id", "site_id", "device_id", "payload"])
            return df.drop_duplicates(subset=["grid_id", "site_id", "device_id"]).reset_index(drop=True)

        site_ids = None
        device_ids = None

        if isinstance(data, dict):
            site_ids = data.get("site_ids") or data.get("siteIds") or data.get("sites")
            device_ids = data.get("device_ids") or data.get("deviceIds") or data.get("devices")

        rows: List[Dict[str, Any]] = []

        if isinstance(site_ids, list):
            for sid in site_ids:
                if sid is None:
                    continue
                rows.append(
                    {
                        "grid_id": grid_id_str,
                        "site_id": str(sid),
                        "device_id": None,
                        "payload": {"source": "generate", "type": "site_id_only"},
                    }
                )

        if isinstance(device_ids, list):
            for did in device_ids:
                if did is None:
                    continue
                rows.append(
                    {
                        "grid_id": grid_id_str,
                        "site_id": None,
                        "device_id": str(did),
                        "payload": {"source": "generate", "type": "device_id_only"},
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["grid_id", "site_id", "device_id", "payload"])

        df = df.drop_duplicates(subset=["grid_id", "site_id", "device_id"]).reset_index(drop=True)
        return df

    except Exception as e:
        raise RuntimeError(f"grid_generate_to_site_devices_df failed for grid_id={grid_id}: {e}") from e


def unique_grid_ids(grids_df: pd.DataFrame) -> List[str]:
    try:
        if grids_df is None or grids_df.empty:
            return []
        if "grid_id" not in grids_df.columns:
            return []
        return sorted(grids_df["grid_id"].dropna().astype(str).unique().tolist())
    except Exception as e:
        raise RuntimeError(f"unique_grid_ids failed: {e}") from e


def filter_grids_by_name(grids_df: pd.DataFrame, needle: str) -> pd.DataFrame:
    """
    Filter grids whose name/long_name contains needle (case-insensitive).
    """
    try:
        if grids_df is None or grids_df.empty:
            return grids_df

        n = (needle or "").strip().lower()
        if not n:
            return grids_df

        def _match(x: Any) -> bool:
            if x is None:
                return False
            return n in str(x).lower()

        name_mask = grids_df["name"].apply(_match) if "name" in grids_df.columns else False
        long_mask = grids_df["long_name"].apply(_match) if "long_name" in grids_df.columns else False

        return grids_df[name_mask | long_mask].copy().reset_index(drop=True)

    except Exception as e:
        raise RuntimeError(f"filter_grids_by_name failed: {e}") from e
