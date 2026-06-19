import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from pyresample import geometry, kd_tree
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import geopandas as gpd

# ===========================
# CONFIGURATION
# ===========================
DATA_ROOT = "/mnt/e/Airquality/Datasets"

# Raw Data Sources
VNP02_ROOT = os.path.join(DATA_ROOT, "AOD/RAW_AOD/VNP02MOD_2-20260501_114005")
VNP03_ROOT = os.path.join(DATA_ROOT, "AOD/RAW_AOD/VNP03MOD_2-20260501_114859")

# UPDATE THIS LINE: Point this to the folder where you downloaded the Cloud Mask
CLDMSK_ROOT = os.path.join(DATA_ROOT, "AOD/RAW_AOD/CLDMSK_L2_VIIRS_SNPP_1-20260506_184021") 

BOUNDARY_PATH = os.path.join(DATA_ROOT, "RAW/BOUNDARY/nairobi_boundary.geojson")

# Output Structure
OUT_INTERIM = os.path.join(DATA_ROOT, "Interim/reprojected_viirs")
OUT_STATS = os.path.join(DATA_ROOT, "Interim/STATS")
OUT_MONTH = os.path.join(DATA_ROOT, "Interim/REL_MONTHLY_MEDIAN")
OUT_DAILY = os.path.join(DATA_ROOT, "Interim/REL_DAILY_FUSED")

os.makedirs(OUT_INTERIM, exist_ok=True)
os.makedirs(OUT_STATS, exist_ok=True)
os.makedirs(OUT_MONTH, exist_ok=True)
os.makedirs(OUT_DAILY, exist_ok=True)

DAILY_STATS_CSV = os.path.join(OUT_STATS, "daily_coverage_cf_stats.csv")

# Target Projection: UTM Zone 37S (Nairobi)
TARGET_CRS = "EPSG:32737"
RESOLUTION = 1000  # VIIRS M-Band nominal resolution in meters

# ===========================
# HELPER FUNCTIONS
# ===========================

def get_scaled_and_filtered_data(ds, band_name, valid_max=65527):
    """Extracts a VIIRS band, applies scaling, and filters it using hardware QA flags."""
    band = ds[band_name]
    scale = band.attrs.get('scale_factor', 1.0)
    offset = band.attrs.get('add_offset', 0.0)
    data = band.values.astype(float)
    
    flag_name = f"{band_name}_quality_flags"
    flags = ds[flag_name].values
    
    # Hardware failures: Out of Range(2), Saturation(4), DG Anomaly(64), 
    # Some Saturation(128), Bowtie(256), Missing EV(512), Cal Fail(1024)
    critical_failures = 2 | 4 | 64 | 128 | 256 | 512 | 1024
    bad_pixel_mask = (flags & critical_failures) != 0
    
    data[data > valid_max] = np.nan
    data[bad_pixel_mask] = np.nan
    
    return (data * scale) + offset

def get_angle_data(ds, angle_name, valid_max=65527):
    """Extracts viewing/solar angles."""
    band = ds[angle_name]
    scale = band.attrs.get('scale_factor', 1.0)
    offset = band.attrs.get('add_offset', 0.0)
    data = band.values.astype(float)
    data[data > valid_max] = np.nan
    return (data * scale) + offset

def process_granule(vnp02_path, vnp03_path, cldmsk_path, nairobi_gdf):
    """Processes a single VIIRS granule to UTM GeoTIFF."""
    try:
        ds02_obs = xr.open_dataset(vnp02_path, group='observation_data')
        
        # Skip Nighttime Granules
        if 'M03' not in ds02_obs.variables:
            return None
            
        ds03_geo = xr.open_dataset(vnp03_path, group='geolocation_data')
        ds_cld = xr.open_dataset(cldmsk_path, group='geophysical_data') 
        
        # 1. Extract Geolocation
        lats = ds03_geo['latitude'].values
        lons = ds03_geo['longitude'].values
        
        # 2. Define Swath and Target Area
        swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
        nairobi_utm = nairobi_gdf.to_crs(TARGET_CRS)
        minx, miny, maxx, maxy = nairobi_utm.total_bounds
        
        area_extent = (minx - 5000, miny - 5000, maxx + 5000, maxy + 5000)
        cols = int((area_extent[2] - area_extent[0]) / RESOLUTION)
        rows = int((area_extent[3] - area_extent[1]) / RESOLUTION)
        
        target_def = geometry.AreaDefinition(
            'nairobi_utm', 'Nairobi UTM 37S', 'utm37s',
            TARGET_CRS, cols, rows, area_extent
        )
        transform = from_bounds(area_extent[0], area_extent[1], area_extent[2], area_extent[3], cols, rows)

        # 3. Extract & Clean Bands (M03, M05, M11) + Angles
        sza_data = get_angle_data(ds03_geo, 'solar_zenith')
        
        bands_to_process = {
            'M03': get_scaled_and_filtered_data(ds02_obs, 'M03'),
            'M05': get_scaled_and_filtered_data(ds02_obs, 'M05'),
            'M11': get_scaled_and_filtered_data(ds02_obs, 'M11'),
            'SZA': sza_data,
            'SAA': get_angle_data(ds03_geo, 'solar_azimuth'),
            'VZA': get_angle_data(ds03_geo, 'sensor_zenith'),
            'VAA': get_angle_data(ds03_geo, 'sensor_azimuth')
        }
        
        # ---------------------------------------------------------
        # APPLY LIU ET AL. (2019) & OFFICIAL CLOUD FILTERS
        # ---------------------------------------------------------
        sza_invalid_mask = (sza_data >= 80.0) | np.isnan(sza_data)
        
        # Extract Official Cloud Mask (Value 3 = Confident Clear)
        cld_mask_data = ds_cld['Integer_Cloud_Mask'].values
        if cld_mask_data.ndim == 3:
            cld_mask_data = cld_mask_data[0, :, :]
        cloudy_pixel_mask = (cld_mask_data != 3)
        
        for band in ['M03', 'M05', 'M11']:
            bands_to_process[band][sza_invalid_mask] = np.nan
            bands_to_process[band][cloudy_pixel_mask] = np.nan
            
            # Physical bounds (0 to 1)
            reflectance_data = bands_to_process[band]
            invalid_reflectance = (reflectance_data <= 0.0) | (reflectance_data >= 1.0)
            bands_to_process[band][invalid_reflectance] = np.nan
        # ---------------------------------------------------------

        # 4. Resample
        resampled_bands = {}
        for name, data in bands_to_process.items():
            resampled_bands[name] = kd_tree.resample_nearest(
                swath_def, data, target_def, 
                radius_of_influence=2000, fill_value=np.nan
            )

        # 5. Calculate Coverage %
        poly_mask = rasterize(
            [(nairobi_utm.geometry.iloc[0], 1)],
            out_shape=(rows, cols),
            transform=transform,
            fill=0, all_touched=True
        )
        roi_pixels = resampled_bands['M03'][poly_mask == 1]
        valid_count = np.count_nonzero(~np.isnan(roi_pixels))
        total_roi = np.count_nonzero(poly_mask == 1)
        coverage_pct = (valid_count / total_roi) * 100 if total_roi > 0 else 0

        # 6. Save TIF
        granule_id = os.path.basename(vnp02_path).split('.')[1] + "_" + os.path.basename(vnp02_path).split('.')[2]
        out_tif = os.path.join(OUT_INTERIM, f"VIIRS_Nairobi_UTM_{granule_id}.tif")
        
        with rasterio.open(
            out_tif, 'w', driver='GTiff',
            height=rows, width=cols, count=len(resampled_bands),
            dtype='float32', crs=TARGET_CRS,
            transform=transform,
            nodata=np.nan
        ) as dst:
            for i, (name, data) in enumerate(resampled_bands.items(), 1):
                dst.write(data.astype('float32'), i)
                dst.set_band_description(i, name)

        return {"granule": granule_id, "coverage_pct": coverage_pct, "path": out_tif}

    except Exception as e:
        print(f"Error processing {os.path.basename(vnp02_path)}: {e}")
        return None
    finally:
        if 'ds02_obs' in locals(): ds02_obs.close()
        if 'ds03_geo' in locals(): ds03_geo.close()
        if 'ds_cld' in locals(): ds_cld.close()

# ===========================
# MAIN BATCH EXECUTION
# ===========================

nairobi_gdf = gpd.read_file(BOUNDARY_PATH)
vnp02_files = sorted(glob.glob(os.path.join(VNP02_ROOT, "VNP02MOD.A*.nc")))
stats_accumulator = []

print(f"Starting batch process for {len(vnp02_files)} granules...")

for f02 in vnp02_files:
    timestamp = ".".join(os.path.basename(f02).split('.')[1:3])
    
    f03_matches = glob.glob(os.path.join(VNP03_ROOT, f"VNP03MOD.{timestamp}*.nc"))
    # Match the Cloud Mask
    cld_matches = glob.glob(os.path.join(CLDMSK_ROOT, f"*.{timestamp}*.nc"))
    
    if not f03_matches or not cld_matches:
        print(f"Skipping {timestamp}: Missing matching Geolocation or Cloud Mask")
        continue
        
    result = process_granule(f02, f03_matches[0], cld_matches[0], nairobi_gdf)
    
    if result:
        stats_accumulator.append(result)
        print(f"Processed {result['granule']} | Coverage: {result['coverage_pct']:.2f}%")

# ===========================
# IN-MEMORY DEDUPLICATION & CLEANUP
# ===========================
print("\n--- Running Deduplication & Cleanup ---")
stats_df = pd.DataFrame(stats_accumulator)

if not stats_df.empty:
    # 1. Create a 'date' column
    stats_df['date'] = stats_df['granule'].apply(lambda x: x.split('_')[0])

    # 2. Find the index of the best image per day
    best_idx = stats_df.groupby('date')['coverage_pct'].idxmax()

    # 3. Identify physical files that were NOT the best
    files_to_delete = stats_df[~stats_df.index.isin(best_idx)]['path'].tolist()

    # 4. Delete the sub-optimal TIFFs from the hard drive
    deleted_count = 0
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            deleted_count += 1
            
    print(f"Deleted {deleted_count} redundant overlapping granules.")

    # 5. Create final clean dataframe and save the CSV
    final_df = stats_df.loc[best_idx].drop(columns=['date']).reset_index(drop=True)
    final_df.to_csv(DAILY_STATS_CSV, index=False)
    
    print(f"Successfully processed {len(final_df)} unique daily scenes.")
    print(f"Metadata saved to: {DAILY_STATS_CSV}")
else:
    print("No granules were successfully processed.")