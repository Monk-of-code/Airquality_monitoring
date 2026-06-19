import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
import warnings

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xarray")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

# ---------------------------
# 1. CONFIGURATION
# ---------------------------
DATA_ROOT = "/mnt/e/Airquality/Datasets"
LAND_DIR = os.path.join(DATA_ROOT, "RAW/ERA5LAND/*.nc")
PBLH_TIF = os.path.join(DATA_ROOT, "RAW/ERA5PBLH/Nairobi_PBLH_2025_composite.tif")
GEOJSON_BOUNDARY = os.path.join(DATA_ROOT, "RAW/BOUNDARY/nairobi_Boundary.geojson")
OUTPUT_DIR = os.path.join(DATA_ROOT, "PROCESSED/DAILY_1KM_MET_STACK")

TARGET_CRS = "EPSG:32737"
TARGET_RES = 1000 
NODATA_VAL = -9999.0

BAND_NAMES = [
    "Relative_Humidity", 
    "U_Wind_10m", 
    "V_Wind_10m", 
    "Temp_2m", 
    "Surface_Pressure", 
    "PBL_Height",
    "Total_Precipitation_mm"
]

def calculate_rh(t2m, d2m):
    """Magnus-Tetens formula (ECMWF standard)."""
    t_c, td_c = t2m - 273.15, d2m - 273.15
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e  = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    return np.clip((e / es) * 100, 0, 100)

# ---------------------------
# 2. RUN PIPELINE
# ---------------------------
def run_production():
    try:
        print(" Loading ERA5-Land and PBLH datasets...")
        ds_land = xr.open_mfdataset(LAND_DIR, combine='by_coords')
        da_pblh_all = rxr.open_rasterio(PBLH_TIF, chunks={'band': 1})

        rename_map = {"valid_time": "time", "latitude": "y", "longitude": "x"}
        ds_land = ds_land.rename({k: v for k, v in rename_map.items() if k in ds_land.coords})
        
        # --- DUAL AGGREGATION STRATEGY ---
        print("Resampling: Mean for states, Max for accumulated precipitation...")
        
        # 1. Daily Mean for atmospheric states
        daily_mean = ds_land[['t2m', 'd2m', 'u10', 'v10', 'sp']].resample(time="1D").mean()
        
        # 2. Daily Max for precipitation (to capture total daily accumulation)
        # Convert meters to millimeters
        daily_precip = (ds_land['tp'].resample(time="1D").max()) * 1000
        
        # Merge products
        daily_met = xr.merge([daily_mean, daily_precip])
        
        print("🗺 Loading boundary and preparing output grid...")
        gdf = gpd.read_file(GEOJSON_BOUNDARY).to_crs(TARGET_CRS)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for i, t in enumerate(daily_met.time.values):
            date_str = pd.to_datetime(t).strftime('%Y-%m-%d')
            
            try:
                # A. Master Reprojection
                subset_met = daily_met.sel(time=t).rio.write_crs("EPSG:4326").rio.reproject(
                    TARGET_CRS, resolution=TARGET_RES, resampling=Resampling.bilinear
                )
                
                # B. PBLH Alignment (Match well-clipped met grid)
                subset_pblh = da_pblh_all.sel(band=i+1).rio.reproject_match(subset_met)

                # C. Final Calculation
                subset_met["rh"] = calculate_rh(subset_met["t2m"], subset_met["d2m"])
                subset_met["blh"] = subset_pblh

                # D. Clip to Nairobi
                clipped = subset_met.rio.clip(gdf.geometry, gdf.crs, drop=True, all_touched=True)

                # E. Stack 7 Bands
                export_stack = clipped[["rh", "u10", "v10", "t2m", "sp", "blh", "tp"]].to_array(dim="band")
                export_stack.rio.write_nodata(NODATA_VAL, inplace=True)
                
                out_path = os.path.join(OUTPUT_DIR, f"Nairobi_Met_{date_str}.tif")
                export_stack.rio.to_raster(out_path, compress="LZW", dtype="float32")
                
                with rasterio.open(out_path, "r+") as dst:
                    dst.descriptions = tuple(BAND_NAMES)
                
                print(f" Exported (Precip-Max Logic): {date_str}")

            except Exception as e:
                print(f" Failed {date_str}: {e}")

        print("\n SUCCESS: All daily 7-band files processed and aligned.")

    except Exception as e:
        print(f" FATAL ERROR: {e}")

if __name__ == "__main__":
    run_production()