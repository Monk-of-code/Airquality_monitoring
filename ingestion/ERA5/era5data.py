import os
import calendar
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import cdsapi

# ---------------------------
# CONFIG
# ---------------------------
YEAR = 2025
MAX_WORKERS = 3

NAIROBI_BBOX = [-1.17, 36.67, -1.46, 37.27]

# WSL path to your E: drive folders
DATA_DIR = "/mnt/e/Airquality/Datasets/RAW"
ERA5_LAND_DIR = os.path.join(DATA_DIR, "ERA5LAND")
ERA5_PBLH_DIR = os.path.join(DATA_DIR, "ERA5PBLH")

# Path for logs on E: drive
LOG_DIR = "/mnt/e/Airquality/Datasets/Logs/ingestionlogs"

VARIABLES_LAND = [
    "2m_dewpoint_temperature",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_precipitation"
]

# ---------------------------
# LOGGING SETUP
# ---------------------------
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "era5_ingestion.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Add a console logger so you can see progress in your WSL terminal too
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

logger = logging.getLogger(__name__)

# ---------------------------
# HELPERS
# ---------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_days(year, month):
    return [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]

def get_hours():
    return [f"{h:02d}:00" for h in range(24)]

# ---------------------------
# RETRY DOWNLOAD
# ---------------------------
def download_with_retry(dataset, request, filename, retries=3):
    for attempt in range(1, retries + 1):
        try:
            if os.path.exists(filename):
                logger.info(f"Skipping existing file: {filename}")
                return

            logger.info(f"Starting download: {os.path.basename(filename)} (Attempt {attempt})")

            client = cdsapi.Client()
            client.retrieve(dataset, request).download(filename)

            logger.info(f"Successfully saved: {filename}")
            return

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed for {filename}: {e}")

            if attempt < retries:
                sleep_time = 10 * attempt
                logger.info(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Critical Failure: {filename} could not be downloaded.")
                raise

# ---------------------------
# DOWNLOAD LOGIC
# ---------------------------
def download_month(month):
    try:
        days = get_days(YEAR, month)
        hours = get_hours()

        ensure_dir(ERA5_LAND_DIR)
        ensure_dir(ERA5_PBLH_DIR)

        land_file = os.path.join(ERA5_LAND_DIR, f"era5_land_{YEAR}_{month:02d}.nc")
        pblh_file = os.path.join(ERA5_PBLH_DIR, f"era5_pblh_{YEAR}_{month:02d}.nc")

        # ERA5-LAND
        land_request = {
            "variable": VARIABLES_LAND,
            "year": str(YEAR),
            "month": f"{month:02d}",
            "day": days,
            "time": hours,
            "area": NAIROBI_BBOX,
            "data_format": "netcdf",
            "download_format": "unarchived"
        }
        download_with_retry("reanalysis-era5-land", land_request, land_file)

        # ERA5 PBLH
        pblh_request = {
            "product_type": "reanalysis",
            "variable": "boundary_layer_height",
            "year": str(YEAR),
            "month": f"{month:02d}",
            "day": days,
            "time": hours,
            "area": NAIROBI_BBOX,
            "data_format": "netcdf",
            "download_format": "unarchived"
        }
        download_with_retry("reanalysis-era5-single-levels", pblh_request, pblh_file)
        
    except Exception as e:
        logger.error(f"Error processing month {month}: {e}")

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    logger.info("--- ERA5 Ingestion Pipeline Started ---")
    logger.info(f"Target Year: {YEAR} | Max Workers: {MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(download_month, range(1, 13))

    logger.info("--- ERA5 Ingestion Pipeline Complete ---")
