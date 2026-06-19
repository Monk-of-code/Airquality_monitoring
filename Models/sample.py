import pandas as pd
import geopandas as gpd
import rasterio
import os
import glob
import numpy as np

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
CSV_INPUT = "/mnt/e/Airquality/Datasets/Processed/pm2.5/dailypm25avgs.csv"
OUTPUT_CSV = "/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_training_dataset.csv"

# Note: Keep U and V Wind components in the model to preserve directional context
ERA5_BANDS = [
    "Relative_Humidity", "U_Wind_10m", "V_Wind_10m", 
    "Temp_2m", "Surface_Pressure", "PBL_Height", "Total_Precipitation_mm"
]
VIIRS_BANDS = ['viirs_M03', 'viirs_M05', 'viirs_M11', 'viirs_SZA', 'viirs_SAA', 'viirs_VZA', 'viirs_VAA']

GASES_DIR = '/mnt/e/Airquality/Datasets/Processed/gases' 
STATIC_DIR = '/mnt/e/Airquality/Datasets/Processed/Static'
NDVI_DIR = os.path.join(STATIC_DIR, 'NDVI_A')
ERA5_DIR = '/mnt/e/Airquality/Datasets/Processed/ERA5_a'
VIIRS_DIR = '/mnt/e/Airquality/Datasets/Processed/VIIRS_Final_Clipped'

STATIC_RASTERS = {
    'elevation': os.path.join(STATIC_DIR, 'elevation.tif'),
    'building_heights': os.path.join(STATIC_DIR, 'building_heights_aligned.tif'),
    'pop_density': os.path.join(STATIC_DIR, 'population_density_aligned.tif'),
    'industry_influence': os.path.join(STATIC_DIR, 'industry_influence_aligned.tif'),
    'transit_influence': os.path.join(STATIC_DIR, 'transit_influence_aligned.tif'),
    'road_density': os.path.join(STATIC_DIR, 'road_density_aligned.tif'),
    'waste_influence': os.path.join(STATIC_DIR, 'waste_influence_aligned.tif')
}

# ==========================================
# 2. HELPER SAMPLING FUNCTION
# ==========================================
def sample_raster(raster_path, xy_coords, indexes=None):
    """Extracts values from a raster at specific (x, y) coordinates."""
    if not os.path.exists(raster_path):
        count = len(indexes) if indexes else 1
        return [np.nan] * len(xy_coords) if count == 1 else [[np.nan]*count for _ in xy_coords]
    
    with rasterio.open(raster_path) as src:
        idx = indexes if indexes else src.indexes
        samples = list(src.sample(xy_coords, indexes=idx))
    
    if len(idx) == 1:
        return [val[0] for val in samples]
    else:
        return samples 

# ==========================================
# 3. LOAD AND PREPARE DATAFRAME
# ==========================================
print("Loading sensor data and resolving coordinate columns...")
df = pd.read_csv(CSV_INPUT)
df['calc_date'] = pd.to_datetime(df['calc_date'])

# FIX 1: Column Sanitization. Ensure we have exactly 'lat' and 'lon' and drop the messy duplicates
if 'lat' not in df.columns and 'latitude' in df.columns:
    df['lat'] = df['latitude']
if 'lon' not in df.columns and 'longitude' in df.columns:
    df['lon'] = df['longitude']

# Drop the messy/empty explicit full-name columns to avoid any confusion in the final CSV
df = df.drop(columns=['latitude', 'longitude'], errors='ignore')

# Drop any rows where we don't have valid coordinates (prevents silent groupby failures)
df = df.dropna(subset=['lat', 'lon'])

print("Projecting to UTM 37S...")
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
gdf = gdf.to_crs("EPSG:32737")
df['utm_x'] = gdf.geometry.x
df['utm_y'] = gdf.geometry.y

# ==========================================
# 4. EXTRACT STATIC VARIABLES
# ==========================================
print("\nExtracting Static Spatial Features...")
all_coords = list(zip(df['utm_x'], df['utm_y']))

for var_name, path in STATIC_RASTERS.items():
    print(f" -> Sampling {var_name}")
    df[var_name] = sample_raster(path, all_coords)

# ==========================================
# 5. EXTRACT DYNAMIC VARIABLES (Grouped by Date)
# ==========================================
print("\nExtracting Dynamic Features (ERA5, NDVI, VIIRS, TROPOMI)...")

dynamic_cols = ['ndvi', 'NO2', 'CO', 'SO2'] + ERA5_BANDS + VIIRS_BANDS
for col in dynamic_cols:
    df[col] = np.nan

grouped = df.groupby('calc_date')

for date, group in grouped:
    indices = group.index
    coords = list(zip(group['utm_x'], group['utm_y']))
    day_of_year = date.timetuple().tm_yday
    
    # --- A. NDVI (Monthly) ---
    month_str = date.strftime('%m') 
    ndvi_path = os.path.join(NDVI_DIR, f"aligned_nairobi_ndvi_2025_{month_str}.tif")
    df.loc[indices, 'ndvi'] = sample_raster(ndvi_path, coords)
    
    # --- B. ERA5 (Daily) ---
    date_str = date.strftime('%Y-%m-%d')
    era5_path = os.path.join(ERA5_DIR, f"aligned_Nairobi_Met_{date_str}.tif")
    if os.path.exists(era5_path):
        era5_samples = sample_raster(era5_path, coords, indexes=list(range(1, len(ERA5_BANDS) + 1)))
        for i, band_name in enumerate(ERA5_BANDS):
            df.loc[indices, band_name] = [sample[i] for sample in era5_samples]
            
    # --- C. VIIRS (Daily Julian) ---
    date_str_viirs = date.strftime('A%Y%j') 
    viirs_search = glob.glob(os.path.join(VIIRS_DIR, f"clipped_VIIRS_Nairobi_UTM_{date_str_viirs}*.tif"))
    if viirs_search:
        viirs_path = viirs_search[0] 
        viirs_samples = sample_raster(viirs_path, coords, indexes=list(range(1, 8)))
        for i, band_name in enumerate(VIIRS_BANDS):
            df.loc[indices, band_name] = [sample[i] for sample in viirs_samples]

    # --- D. TROPOMI (365-Band Files) ---
    gas_mapping = {'NO2': 'NO2_umol_m2.tif', 'CO': 'CO_mmol_m2.tif', 'SO2': 'SO2_umol_m2.tif'}
    for var, filename in gas_mapping.items():
        gas_path = os.path.join(GASES_DIR, filename)
        df.loc[indices, var] = sample_raster(gas_path, coords, indexes=[day_of_year])

# ==========================================
# 6. PHYSICS & GENERAL FEATURE ENGINEERING
# ==========================================
print("\nEngineering Base Physics Features...")

df['month'] = df['calc_date'].dt.month
df['day_of_week'] = df['calc_date'].dt.dayofweek

df['Wind_Speed_10m'] = np.sqrt(df['U_Wind_10m']**2 + df['V_Wind_10m']**2)

for col in ['viirs_SAA', 'viirs_VAA']:
    df[f'{col}_sin'] = np.sin(np.radians(df[col]))
    df[f'{col}_cos'] = np.cos(np.radians(df[col]))

# ==========================================
# 7. METEOROLOGICAL TIME-SERIES LAGS 
# ==========================================
print("\nResolving multi-device site duplicates using hour counts...")

max_hours = df.groupby(['lon', 'lat', 'calc_date'])['hours_count'].transform('max')
df_filtered = df[df['hours_count'] == max_hours].copy()

print(" -> Merging tied entries via numerical averaging...")
agg_dict = {}
for col in df_filtered.columns:
    if col in ['lon', 'lat', 'calc_date']:
        continue
    if pd.api.types.is_numeric_dtype(df_filtered[col]):
        agg_dict[col] = 'mean'
    else:
        agg_dict[col] = 'first'

df_deduped = df_filtered.groupby(['lon', 'lat', 'calc_date'], as_index=False, dropna=True).agg(agg_dict)

print(" -> Constructing spatial group timelines and lags...")

def compute_true_calendar_metrics(group):
    # FIX 2: Extract coordinates securely from the group's metadata tuple, bypassing the missing columns
    lon_cache, lat_cache = group.name
    
    original_dates = group['calc_date'].copy()
    
    group = group.set_index('calc_date')
    full_date_range = pd.date_range(start=group.index.min(), end=group.index.max(), freq='D')
    group = group.reindex(full_date_range)
    
    group['PBL_Height_lag1'] = group['PBL_Height'].shift(1)
    group['Precip_3d_sum'] = group['Total_Precipitation_mm'].rolling(window=3, min_periods=1).sum()
    group['Relative_Humidity_lag1'] = group['Relative_Humidity'].shift(1)
    
    group = group.loc[original_dates]
    group['calc_date'] = original_dates.values
    
    # Force the coordinates back into the dataframe before returning
    group['lon'] = lon_cache
    group['lat'] = lat_cache
    
    return group

df_deduped = df_deduped.sort_values(by=['lon', 'lat', 'calc_date']).reset_index(drop=True)

# FIX 3: Removed `as_index=False` to guarantee `group.name` is populated correctly as a (lon, lat) tuple
df = df_deduped.groupby(['lon', 'lat'], group_keys=False).apply(compute_true_calendar_metrics).reset_index(drop=True)

# ==========================================
# 8. CLEANUP AND SAVE
# ==========================================
print("\nFinalizing cleanup and exporting dataset...")

lag_cols = ['PBL_Height_lag1', 'Precip_3d_sum', 'Relative_Humidity_lag1']
df = df.dropna(subset=['daily_avg_pm25'] + lag_cols)

# Drop computational helper variables and raw angular dependencies
df = df.drop(columns=['utm_x', 'utm_y', 'viirs_SAA', 'viirs_VAA'])

# Final check to organize columns cleanly
cols = ['device_name', 'site_id', 'site_name', 'calc_date', 'lat', 'lon'] 
cols += [c for c in df.columns if c not in cols]
df = df[cols]

# Export to target workspace directory
df.to_csv(OUTPUT_CSV, index=False)
print(f"Success! Training dataset with strict meteorological lags saved to: {OUTPUT_CSV}")