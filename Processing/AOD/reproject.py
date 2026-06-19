import os
import glob
import logging
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ---------------------------
# CONFIG
# ---------------------------
DATA_ROOT  = "/mnt/e/Airquality/Datasets"
INPUT_DIR   = os.path.join(DATA_ROOT, "AOD/RAW_AOD/MCD19A2_061-20260419_230538")
OUTPUT_DIR  = os.path.join(DATA_ROOT, "INTERIM/reprojected")
LOG_FILE     = os.path.join(DATA_ROOT, "Logs/reprojectionlogs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_FILE, exist_ok=True)

TARGET_CRS = "EPSG:32737"

# ---------------------------
# LOGGING
# ---------------------------

logging.basicConfig(
    filename=os.path.join(LOG_FILE, "reprojectstats.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def log(msg, level="info"):
    print(msg)
    if level == "error":
        logger.error(msg)
    else:
        logger.info(msg)


# ---------------------------
# REPROJECT FUNCTION
# ---------------------------
def reproject_file(hdf_file):

    try:
        aod_path = f'HDF4_EOS:EOS_GRID:"{hdf_file}":grid1km:Optical_Depth_055'
        qa_path  = f'HDF4_EOS:EOS_GRID:"{hdf_file}":grid1km:AOD_QA'

        with rasterio.open(aod_path) as aod_src, rasterio.open(qa_path) as qa_src:

            transform, width, height = calculate_default_transform(
                aod_src.crs,
                TARGET_CRS,
                aod_src.width,
                aod_src.height,
                *aod_src.bounds
            )

            profile = aod_src.profile.copy()
            profile.update({
                "crs": TARGET_CRS,
                "transform": transform,
                "width": width,
                "height": height,
                "count": aod_src.count + qa_src.count,  # combine bands
                "dtype": "float32",
                "nodata": np.nan,
                "driver": "GTiff",
                "compress": "lzw"
            })

            out_name = os.path.basename(hdf_file).replace(".hdf", "_utm37s.tif")
            out_path = os.path.join(OUTPUT_DIR, out_name)

            with rasterio.open(out_path, "w", **profile) as dst:

                band_index = 1

                # --- AOD (bilinear) ---
                for i in range(1, aod_src.count + 1):

                    reproject(
                        source=rasterio.band(aod_src, i),
                        destination=rasterio.band(dst, band_index),
                        src_transform=aod_src.transform,
                        src_crs=aod_src.crs,
                        dst_transform=transform,
                        dst_crs=TARGET_CRS,
                        resampling=Resampling.bilinear
                    )
                    band_index += 1

                # --- QA (nearest neighbour) ---
                for i in range(1, qa_src.count + 1):

                    reproject(
                        source=rasterio.band(qa_src, i),
                        destination=rasterio.band(dst, band_index),
                        src_transform=qa_src.transform,
                        src_crs=qa_src.crs,
                        dst_transform=transform,
                        dst_crs=TARGET_CRS,
                        resampling=Resampling.nearest
                    )
                    band_index += 1

        log(f"✔ Reprojected AOD + QA: {out_name}")

    except Exception as e:
        log(f"❌ Failed: {hdf_file} | {e}")

# ---------------------------
# MAIN
# ---------------------------
def main():

    files = glob.glob(os.path.join(INPUT_DIR, "**", "*.hdf"), recursive=True)

    if not files:
        raise RuntimeError("No HDF files found")

    log(f"Found {len(files)} files")

    for f in files:
        reproject_file(f)

    log("🎉 Reprojection complete (AOD + QA preserved)")


if __name__ == "__main__":
    main()