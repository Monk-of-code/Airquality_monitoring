import asyncio
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import aiohttp


@dataclass(frozen=True)
class AirQoConfig:
    base_url: str = "https://api.airqo.net/api/v2"
    timeout_s: int = 40
    max_retries: int = 4
    backoff_base_s: float = 0.7
    backoff_jitter_s: float = 0.3


def build_grids_summary_url(cfg: AirQoConfig) -> str:
    return f"{cfg.base_url}/devices/grids/summary"


def build_grid_generate_url(grid_id: str, cfg: AirQoConfig) -> str:
    return f"{cfg.base_url}/devices/grids/{grid_id}/generate"


def is_retryable_status(status: int) -> bool:
    return status in (408, 429, 500, 502, 503, 504)


async def fetch_json_with_retries(
    session: aiohttp.ClientSession,
    url: str,
    params: Dict[str, Any],
    cfg: AirQoConfig,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None

    for attempt in range(1, cfg.max_retries + 1):
        try:
            async with session.get(url, params=params, headers=headers) as resp:
                text = await resp.text()

                try:
                    data = await resp.json(content_type=None)
                except Exception:
                    raise RuntimeError(
                        f"Non-JSON response. status={resp.status} "
                        f"content_type={resp.headers.get('Content-Type')} "
                        f"body_head={text[:300]!r}"
                    )

                if resp.status == 200:
                    return data

                err = RuntimeError(
                    f"HTTP {resp.status}. url={url}, message={data.get('message')}, params={params}"
                )

                if is_retryable_status(resp.status) and attempt < cfg.max_retries:
                    last_err = err
                else:
                    raise err

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_err = e

        if attempt < cfg.max_retries:
            backoff = cfg.backoff_base_s * (2 ** (attempt - 1))
            backoff += random.uniform(0, cfg.backoff_jitter_s)
            await asyncio.sleep(backoff)

    raise RuntimeError(f"Failed after {cfg.max_retries} attempts. last_err={last_err}")


async def get_grids_summary_all(
    cfg: AirQoConfig = AirQoConfig(),
    page_size: int = 100,
    tenant: str = "airqo",
    detail_level: str = "summary",
) -> Dict[str, Any]:
    """
    Fetches ALL grids from /devices/grids/summary via pagination.
    Merges into a single response with one "grids" list.
    """
    url = build_grids_summary_url(cfg)

    all_grids: List[Dict[str, Any]] = []
    meta_last: Optional[Dict[str, Any]] = None

    timeout = aiohttp.ClientTimeout(total=cfg.timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        skip = 0
        while True:
            params = {
                "limit": page_size,
                "skip": skip,
                "tenant": tenant,
                "detailLevel": detail_level,
            }

            raw = await fetch_json_with_retries(session, url, params, cfg)

            if not raw.get("success", False):
                raise RuntimeError(f"success=false: {raw.get('message')} error={raw.get('error')}")

            grids = raw.get("grids") or []
            meta = raw.get("meta") or {}
            meta_last = meta or meta_last

            all_grids.extend(grids)

            # robust stop condition using meta.total (preferred)
            total = meta.get("total")
            limit = meta.get("limit", page_size)
            cur_skip = meta.get("skip", skip)

            if total is not None:
                # next page starts at skip + limit
                next_skip = int(cur_skip) + int(limit)
                if next_skip >= int(total):
                    break
                skip = next_skip
                continue

            # fallback stop condition if total missing
            if len(grids) < page_size:
                break

            skip += page_size

    return {"success": True, "message": "merged", "meta": meta_last, "grids": all_grids}


async def get_grid_generate(
    grid_id: str,
    token: str,
    cfg: AirQoConfig = AirQoConfig(),
) -> Dict[str, Any]:
    url = build_grid_generate_url(grid_id, cfg)
    params = {"token": token}

    timeout = aiohttp.ClientTimeout(total=cfg.timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        raw = await fetch_json_with_retries(session, url, params, cfg)

    if not raw.get("success", False):
        raise RuntimeError(f"success=false: {raw.get('message')} error={raw.get('error')}")
    return raw

