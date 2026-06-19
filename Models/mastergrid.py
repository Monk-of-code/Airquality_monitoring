import os
import glob
import warnings
import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
from sklearn.neighbors import BallTree
import geopandas as gpd

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURATION & STRICT PATH AUDITING
# =====================================================================
STAGE1_MODEL_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stg1_xgboost_v3.json'
STAGE2_MODEL_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stg2_final_model.json'
FINAL_MASTER_GRID_CSV = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_full_grid_stage2_2025.csv'

STATIC_DIR = '/mnt/e/Airquality/Datasets/Processed/Static'
NDVI_DIR = os.path.join(STATIC_DIR, 'NDVI_A')
ERA5_DIR = '/mnt/e/Airquality/Datasets/Processed/ERA5_a'
VIIRS_DIR = '/mnt/e/Airquality/Datasets/Processed/VIIRS_Final_Clipped'
GASES_DIR = '/mnt/e/Airquality/Datasets/Processed/gases'

# Locked dictionary containing all active baseline and engineered buffers
STATIC_RASTERS = {
    'elevation': os.path.join(STATIC_DIR, 'elevation.tif'),
    'building_heights': os.path.join(STATIC_DIR, 'building_heights_aligned.tif'),
    'pop_density': os.path.join(STATIC_DIR, 'population_density_aligned.tif'),
    'industry_influence': os.path.join(STATIC_DIR, 'industry_influence_aligned.tif'),
    'transit_influence': os.path.join(STATIC_DIR, 'transit_influence_aligned.tif'),
    'road_density': os.path.join(STATIC_DIR, 'road_density_aligned.tif'),
    'waste_influence': os.path.join(STATIC_DIR, 'waste_influence_aligned.tif')
}

PHYSICAL_ZERO_FEATS = [
    'building_heights', 'pop_density', 'industry_influence', 
    'transit_influence', 'road_density', 'waste_influence'
]
ERA5_BANDS = [
    "Relative_Humidity", "U_Wind_10m", "V_Wind_10m", "Temp_2m", 
    "Surface_Pressure", "PBL_Height", "Total_Precipitation_mm"
]
VIIRS_BANDS = [
    'viirs_M03', 'viirs_M05', 'viirs_M11', 'viirs_SZA', 
    'viirs_SAA', 'viirs_VZA', 'viirs_VAA'
]

print("Loading Stage 1 and Stage 2 production models...")
booster_stg1 = xgb.Booster()
booster_stg1.load_model(STAGE1_MODEL_PATH)
stg1_features = booster_stg1.feature_names

model_stg2 = xgb.XGBRegressor()
model_stg2.load_model(STAGE2_MODEL_PATH)
stg2_features = model_stg2.get_booster().feature_names

# =====================================================================
# 2. MASTER GRID SKELETON EXTRACTION
# =====================================================================
print("\nExtracting verified Nairobi polygon skeleton...")
with rasterio.open(STATIC_RASTERS['elevation']) as src:
    width, height = src.width, src.height
    transform = src.transform
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    u_x, u_y = rasterio.transform.xy(transform, rows, cols, offset='center')
    
    data = src.read(1).flatten()
    nodata_val = src.nodata
    
    if nodata_val is None or np.isnan(nodata_val):
        valid_mask = ~np.isnan(data)
    else:
        valid_mask = (data != nodata_val) & ~np.isnan(data)

master_pixels = pd.DataFrame({
    'utm_x': np.array(u_x).flatten()[valid_mask], 
    'utm_y': np.array(u_y).flatten()[valid_mask]
}).reset_index(drop=True)

# Project to geographic coordinates (EPSG:4326) ensuring specific column names
gdf_pixels = gpd.GeoDataFrame(
    master_pixels, 
    geometry=gpd.points_from_xy(master_pixels.utm_x, master_pixels.utm_y), 
    crs="EPSG:32737"
).to_crs("EPSG:4326")

master_pixels['lon'] = gdf_pixels.geometry.x.values
master_pixels['lat'] = gdf_pixels.geometry.y.values

# Pre-sample all static layers
pixel_coords = list(zip(master_pixels['utm_x'], master_pixels['utm_y']))
for var, path in STATIC_RASTERS.items():
    if os.path.exists(path):
        with rasterio.open(path) as src:
            samples = [v[0] for v in list(src.sample(pixel_coords))]
            master_pixels[var] = np.nan_to_num(samples, nan=0.0) if var in PHYSICAL_ZERO_FEATS else samples
    else:
        master_pixels[var] = 0.0

print(f"Locked baseline skeleton: exactly {len(master_pixels)} verified grid pixels.")

# =====================================================================
# 3. FULL 365-DAY SPATIOTEMPORAL LOOP
# =====================================================================
print("\nInitiating full 2025 inference loop...")
all_days_records = []
date_range = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')

EARTH_RADIUS_KM = 6371.0

# Initialize state buffers outside the loop to carry memory across days
state_buffer = {
    'PBL_Height_lag1': None,
    'Relative_Humidity_lag1': None,
    'Precip_history': []
}

for idx, dt in enumerate(date_range, 1):
    doy, month_str, date_str = dt.dayofyear, dt.strftime('%m'), dt.strftime('%Y-%m-%d')
    
    if idx % 25 == 0 or idx == 1 or idx == 365:
        print(f" -> Processing Day {idx:03d}/365: {date_str}...")
        
    df_grid = master_pixels.copy()
    df_grid['calc_date'] = date_str
    df_grid['month'] = dt.month
    df_grid['day_of_week'] = dt.dayofweek
    df_grid['t1_n'] = np.cos(2 * np.pi * doy / 365.0 - np.pi / 12)
    df_grid['t2_n'] = np.sin(2 * np.pi * doy / 365.0 - np.pi / 12)
    df_grid['t3']   = doy / 365.0
    
    # --- A. Dynamic Raster Sampling ---
    ndvi_path = os.path.join(NDVI_DIR, f"aligned_nairobi_ndvi_2025_{month_str}.tif")
    df_grid['ndvi'] = np.nan
    if os.path.exists(ndvi_path):
        with rasterio.open(ndvi_path) as src: 
            df_grid['ndvi'] = [v[0] for v in src.sample(pixel_coords)]
        
    era5_path = os.path.join(ERA5_DIR, f"aligned_Nairobi_Met_{date_str}.tif")
    for b in ERA5_BANDS: df_grid[b] = np.nan
    if os.path.exists(era5_path):
        with rasterio.open(era5_path) as src:
            samples = list(src.sample(pixel_coords, indexes=list(range(1, 8))))
            for i, b in enumerate(ERA5_BANDS): 
                df_grid[b] = [s[i] for s in samples]

    v_search = glob.glob(os.path.join(VIIRS_DIR, f"clipped_VIIRS_Nairobi_UTM_A2025{dt.strftime('%j')}*.tif"))
    for b in VIIRS_BANDS: df_grid[b] = np.nan
    if v_search:
        with rasterio.open(v_search[0]) as src:
            samples = list(src.sample(pixel_coords, indexes=list(range(1, 8))))
            for i, b in enumerate(VIIRS_BANDS): 
                df_grid[b] = [s[i] for s in samples]

    for var, fname in {'NO2': 'NO2_umol_m2.tif', 'CO': 'CO_mmol_m2.tif', 'SO2': 'SO2_umol_m2.tif'}.items():
        g_path = os.path.join(GASES_DIR, fname)
        df_grid[var] = np.nan
        if os.path.exists(g_path):
            with rasterio.open(g_path) as src: 
                if src.count >= doy:
                    df_grid[var] = [v[0] for v in src.sample(pixel_coords, indexes=[doy])]

    # --- B. Physics & Angular Transformations ---
    df_grid['Wind_Speed_10m'] = np.sqrt(df_grid['U_Wind_10m']**2 + df_grid['V_Wind_10m']**2)
    for col in ['viirs_SAA', 'viirs_VAA']:
        df_grid[f'{col}_sin'] = np.sin(np.radians(df_grid[col]))
        df_grid[f'{col}_cos'] = np.cos(np.radians(df_grid[col]))

    df_grid['st_interact_1'] = df_grid['lat'] * df_grid['t1_n']
    df_grid['st_interact_2'] = df_grid['lon'] * df_grid['t2_n']

    # --- C. Memory & Lag Feature Generation ---
    if state_buffer['PBL_Height_lag1'] is not None:
        df_grid['PBL_Height_lag1'] = state_buffer['PBL_Height_lag1']
        df_grid['Relative_Humidity_lag1'] = state_buffer['Relative_Humidity_lag1']
    else:
        df_grid['PBL_Height_lag1'] = df_grid['PBL_Height']
        df_grid['Relative_Humidity_lag1'] = df_grid['Relative_Humidity']
        
    state_buffer['Precip_history'].append(df_grid['Total_Precipitation_mm'].fillna(0).values.copy())
    if len(state_buffer['Precip_history']) > 3:
        state_buffer['Precip_history'].pop(0) 
    
    df_grid['Precip_3d_sum'] = np.sum(state_buffer['Precip_history'], axis=0)
    
    state_buffer['PBL_Height_lag1'] = df_grid['PBL_Height'].values.copy()
    state_buffer['Relative_Humidity_lag1'] = df_grid['Relative_Humidity'].values.copy()

    # --- D. Stage 1 Background Field Inference ---
    dmatrix_stg1 = xgb.DMatrix(df_grid[stg1_features].astype('float32'))
    
    # FIXED: Direct prediction in physical space. No np.expm1().
    df_grid['stg1_pm25_physical'] = booster_stg1.predict(dmatrix_stg1).astype('float32')
    
    # Alias the feature exactly as Stage 2 expects it
    df_grid['pm25_stage1_pred_physical_OOF'] = df_grid['stg1_pm25_physical']

    # --- E. Stage 2 Spatial Lag (Ps) using Adaptive Bandwidth (k=3) ---
    grid_coords_rad = np.radians(df_grid[['lat', 'lon']].values)
    grid_stg1_vals = df_grid['stg1_pm25_physical'].values
    
    # Query k=4 (0th is self, next 3 are the nearest neighbors)
    tree = BallTree(grid_coords_rad, metric='haversine')
    dists, idxs = tree.query(grid_coords_rad, k=4)
    
    # Isolate the 3 neighbors (ignoring the self-pixel at index 0)
    neighbor_dists = dists[:, 1:] * EARTH_RADIUS_KM
    neighbor_vals = grid_stg1_vals[idxs[:, 1:]]
    
    # Vectorized computation of adaptive bandwidth Gaussian kernel
    h_i = np.maximum(neighbor_dists[:, -1], 0.001) # Dist to farthest neighbor
    
    # Calculate weights: exp(-(dist^2) / (h_i^2))
    weights = np.exp(-(neighbor_dists**2) / (h_i[:, np.newaxis]**2))
    w_sum = np.sum(weights, axis=1)
    
    # Calculate weighted average Ps. Fallback to arithmetic mean if weight sum is 0
    weighted_sum = np.sum(weights * neighbor_vals, axis=1)
    ps_column = np.where(w_sum > 0, weighted_sum / w_sum, np.mean(neighbor_vals, axis=1))
    
    df_grid['improved_Ps'] = ps_column.astype('float32')

    # --- F. Stage 2 Correction Inference ---
    X_stg2 = df_grid[stg2_features].astype('float32')
    
    # FIXED: Direct prediction in physical space. No np.expm1().
    df_grid['final_pm25_physical'] = model_stg2.predict(X_stg2).astype('float32')

    # Keep memory clean by retaining only essential mapping fields
    export_fields = ['calc_date', 'lat', 'lon', 'stg1_pm25_physical', 'improved_Ps', 'final_pm25_physical']
    all_days_records.append(df_grid[export_fields])

# Efficient end-of-loop concatenation avoids pandas memory fragmentation
print("\nAssembling full master dataset...")
df_final_master_grid = pd.concat(all_days_records, ignore_index=True)
os.makedirs(os.path.dirname(FINAL_MASTER_GRID_CSV), exist_ok=True)
df_final_master_grid.to_csv(FINAL_MASTER_GRID_CSV, index=False)

print("\n" + "="*70)
print(f" SUCCESS: Full 2025 Master Grid assembled successfully.")
print(f"Total Spatiotemporal Records Exported: {len(df_final_master_grid)}")
print(f"Master CSV Saved to: {FINAL_MASTER_GRID_CSV}")
print("="*70)