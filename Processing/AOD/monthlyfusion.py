import os
import glob
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import logging
from rasterio.mask import mask
from collections import defaultdict

# ===========================
# CONFIG
# ===========================
DATA_ROOT = "/mnt/e/Airquality/Datasets"

INPUT_DIR = os.path.join(DATA_ROOT, "INTERIM/reprojected")
BOUNDARY  = os.path.join(DATA_ROOT, "RAW/BOUNDARY/nairobi_boundary.geojson")

OUT_STATS = os.path.join(DATA_ROOT, "INTERIM/STATS")
OUT_MONTH = os.path.join(DATA_ROOT, "INTERIM/REL_MONTHLY_MEDIAN")
OUT_DAILY = os.path.join(DATA_ROOT, "INTERIM/REL_DAILY_FUSED")

os.makedirs(OUT_STATS, exist_ok=True)
os.makedirs(OUT_MONTH, exist_ok=True)
os.makedirs(OUT_DAILY, exist_ok=True)

DAILY_CSV   = os.path.join(OUT_STATS, "daily_REL_stats.csv")
MONTHLY_CSV = os.path.join(OUT_STATS, "monthly_REL_stats.csv")

# ===========================
# LOGGING
# ===========================
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

gdf = gpd.read_file(BOUNDARY)

# ===========================
# DATE PARSER
# ===========================
def extract_date(fname):
    try:
        date_str = fname.split(".A")[1][:7]
        return pd.to_datetime(date_str, format="%Y%j")
    except:
        return None

# ===========================
# QA FILTER (MODERATE)
# ===========================
def qa_filter(aod, qa):

    cloud = qa & 0x7
    adj   = (qa >> 5) & 0x7

    cloud_ok = (cloud == 1) | (cloud == 2) | (cloud == 3)
    adj_ok   = (adj == 0) | (adj == 1) | (adj ==2)

    mask = cloud_ok & adj_ok

    return np.where(mask, aod, np.nan), cloud, adj


# ===========================
# QA WEIGHTS
# ===========================
def qa_weights(cloud, adj):

    w_cloud = np.select(
        [cloud == 1,cloud == 2, cloud ==3],
        [1.0 , 0.85 , 0.60],
        default=0.0
    )

    w_adj = np.select(
        [adj == 0, adj == 1 , adj == 2],
        [1.0, 0.85, 0.60],
        default=0.0
    )

    return w_cloud * w_adj


# ===========================
# CLIP FUNCTION
# ===========================
def clip(path):
    with rasterio.open(path) as src:

        gdf_proj = gdf.to_crs(src.crs)

        out_img, out_transform = mask(
            src,
            gdf_proj.geometry,
            crop=True,
            filled=True,
            nodata=np.nan
        )

        profile = src.profile.copy()
        profile.update({
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_transform,
            "nodata": np.nan
        })

    return out_img, profile


# ===========================
# GROUP BY DAY
# ===========================
files = glob.glob(os.path.join(INPUT_DIR, "*.tif"))

day_groups = defaultdict(list)

for f in files:
    d = extract_date(os.path.basename(f))
    if d is not None:
        day_groups[d].append(f)

logging.info(f"Found {len(day_groups)} unique days")


# ===========================
# DAILY FUSION (QA-WEIGHTED)
# ===========================
daily_records = []

for date, flist in sorted(day_groups.items()):

    weighted_sum = None
    weight_sum = None

    raw_cov_list = []
    valid_after = 0
    base_profile = None

    for f in flist:

        img, profile = clip(f)

        if base_profile is None:
            base_profile = profile

        aod = img[0].astype(np.float32)
        qa  = np.nan_to_num(img[1], nan=0).astype(np.uint16)

        # BEFORE QC
        raw_cov_list.append(np.isfinite(aod).mean() * 100)

        # APPLY QA FILTER
        aod_qc, cloud, adj = qa_filter(aod, qa)

        weights = qa_weights(cloud, adj)

        valid = np.isfinite(aod_qc) & (weights > 0)

        if np.sum(valid) == 0:
            continue

        aod_w = np.where(valid, aod_qc * weights, 0)
        w = np.where(valid, weights, 0)

        if weighted_sum is None:
            weighted_sum = aod_w
            weight_sum = w
        else:
            weighted_sum += aod_w
            weight_sum += w

        valid_after += 1

    if weight_sum is None or np.sum(weight_sum) == 0:
        continue

    # FINAL DAILY FUSED PRODUCT
    daily_fused = weighted_sum / np.where(weight_sum == 0, np.nan, weight_sum)

    # SAVE DAILY
    out_daily = os.path.join(OUT_DAILY, f"AOD_{date.strftime('%Y-%m-%d')}.tif")

    base_profile.update({"count": 1, "dtype": "float32", "nodata": np.nan})

    with rasterio.open(out_daily, "w", **base_profile) as dst:
        dst.write(daily_fused.astype(np.float32), 1)

    daily_cov_after = np.isfinite(daily_fused).mean() * 100

    daily_records.append({
        "date": date,
        "n_files": len(flist),
        "valid_files_after_qc": valid_after,
        "coverage_before_qc": np.mean(raw_cov_list),
        "coverage_after_qc": daily_cov_after
    })

    logging.info(f"{date.date()} | after_QC={daily_cov_after:.2f}% | files={len(flist)}")


# ===========================
# DAILY DF
# ===========================
daily_df = pd.DataFrame(daily_records)
daily_df["month"] = daily_df["date"].dt.to_period("M")
daily_df.to_csv(DAILY_CSV, index=False)


# ===========================
# MONTHLY COMPOSITE (FROM DAILY FUSED)
# ===========================
monthly_records = []

for month, group in daily_df.groupby("month"):

    stack = []

    for d in group["date"]:
        f = os.path.join(OUT_DAILY, f"AOD_{d.strftime('%Y-%m-%d')}.tif")
        if os.path.exists(f):
            with rasterio.open(f) as src:
                stack.append(src.read(1))

    if len(stack) == 0:
        continue

    stack = np.stack(stack)

    monthly_median = np.nanmedian(stack, axis=0)

    monthly_cov = np.isfinite(monthly_median).mean() * 100
    mean_daily_cov = group["coverage_after_qc"].mean()
    gain = monthly_cov - mean_daily_cov

    out_tif = os.path.join(OUT_MONTH, f"AOD_{month}_median.tif")

    profile.update({"count": 1, "dtype": "float32", "nodata": np.nan})

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(monthly_median.astype(np.float32), 1)

    monthly_records.append({
        "month": str(month),
        "days_recorded": len(group),
        "valid_days": group["valid_files_after_qc"].sum(),
        "coverage_before_qc": group["coverage_before_qc"].mean(),
        "coverage_after_qc": mean_daily_cov,
        "monthly_coverage": monthly_cov,
        "coverage_gain": gain
    })

    logging.info(f"{month} | days={len(group)} | gain={gain:.2f}%")


# ===========================
# SAVE MONTHLY
# ===========================
pd.DataFrame(monthly_records).to_csv(MONTHLY_CSV, index=False)

logging.info("PIPELINE COMPLETE")