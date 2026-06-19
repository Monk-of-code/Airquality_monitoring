import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from request import get_grids_summary_all, get_grid_generate
from transform import (
    grids_to_df,
    grids_summary_to_sites_df,
    filter_grids_by_name,
    grid_generate_to_site_devices_df,
)
from load import (
    PostgresConfig,
    build_sqlalchemy_url_from_env,
    upsert_grids,
    upsert_sites,
    upsert_site_devices,
)


def load_env_safely():
    # walk upwards until you find .env
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        candidate = p / ".env"
        if candidate.exists():
            load_dotenv(dotenv_path=candidate)
            return str(candidate)
    return None


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


async def run_metadata():
    env_path = load_env_safely()
    print("env loaded from:", env_path)

    token = _require_env("AIRQO_API_TOKEN")

    # Build ONE config (tables are now part of config)
    cfg = PostgresConfig(
        sqlalchemy_url=build_sqlalchemy_url_from_env(),
        schema="bronze",
        grids_table="airqo_metadata_grids",
        sites_table="airqo_metadata_sites",
        site_devices_table="airqo_site_devices",  # keep your existing name if you want
    )

    # 1) summary (ALL pages)
    summary_all = await get_grids_summary_all(page_size=100)
    grids_df = grids_to_df(summary_all)
    sites_df = grids_summary_to_sites_df(summary_all)

    print("grids:", len(grids_df))
    print("sites:", len(sites_df))

    print("grids upserted:", upsert_grids(grids_df, cfg))
    print("sites upserted:", upsert_sites(sites_df, cfg))

    # 2) optional: Nairobi only, then generate device ids
    nairobi = filter_grids_by_name(grids_df, "nairobi")
    print("nairobi grids found:", len(nairobi))

    total_mapping = 0
    for grid_id in nairobi["grid_id"].astype(str).tolist():
        gen_raw = await get_grid_generate(grid_id=grid_id, token=token)
        mapping_df = grid_generate_to_site_devices_df(gen_raw, grid_id=grid_id)

        total_mapping += upsert_site_devices(mapping_df, cfg)
        print(f"[{grid_id}] mapping rows:", len(mapping_df))

    print("total mapping upserted:", total_mapping)


def main():
    try:
        asyncio.run(run_metadata())
    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    main()