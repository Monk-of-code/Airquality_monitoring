import os
import glob
import warnings
import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
from sklearn.neighbors import BallTree
import geopandas as gpd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURATION & UPFRONT DATA INTEGRITY
# =====================================================================
print("Initializing Connected Spatiotemporal Pipeline (Stage 2 CV Optimization)...")

PRIMARY_CSV = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stg1_results_v3.csv'
STAGE1_MODEL_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stg1_xgboost_v3.json'
OUTPUT_STAGE2_CSV = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stage2_training.csv'

DAILY_INSPECTION_DIR = '/mnt/e/Airquality/Datasets/Processed/pm2.5/stage1_daily_predictions/'
os.makedirs(DAILY_INSPECTION_DIR, exist_ok=True)

STATIC_DIR = '/mnt/e/Airquality/Datasets/Processed/Static'
NDVI_DIR = os.path.join(STATIC_DIR, 'NDVI_A')
ERA5_DIR = '/mnt/e/Airquality/Datasets/Processed/ERA5_a'
VIIRS_DIR = '/mnt/e/Airquality/Datasets/Processed/VIIRS_Final_Clipped'
GASES_DIR = '/mnt/e/Airquality/Datasets/Processed/gases'

STATIC_RASTERS = {
    'elevation': os.path.join(STATIC_DIR, 'elevation.tif'),
    'building_heights': os.path.join(STATIC_DIR, 'building_heights_aligned.tif'),
    'pop_density': os.path.join(STATIC_DIR, 'population_density_aligned.tif'),
    'industry_influence': os.path.join(STATIC_DIR, 'industry_influence_aligned.tif'),
    'transit_influence': os.path.join(STATIC_DIR, 'transit_influence_aligned.tif'),
    'road_density': os.path.join(STATIC_DIR, 'road_density_aligned.tif'),
    'waste_influence': os.path.join(STATIC_DIR, 'waste_influence_aligned.tif')
}

PHYSICAL_ZERO_FEATS = ['building_heights', 'pop_density', 'industry_influence', 'transit_influence', 'road_density','waste_influence']
ERA5_BANDS = ["Relative_Humidity", "U_Wind_10m", "V_Wind_10m", "Temp_2m", "Surface_Pressure", "PBL_Height", "Total_Precipitation_mm"]
VIIRS_BANDS = ['viirs_M03', 'viirs_M05', 'viirs_M11', 'viirs_SZA', 'viirs_SAA', 'viirs_VZA', 'viirs_VAA']

# --- Load Base Station Data ---
df_stations = pd.read_csv(PRIMARY_CSV)
df_stations['calc_date'] = pd.to_datetime(df_stations['calc_date'])

# --- Clean and Align Target Observations ---
df_stations = df_stations[(df_stations['calc_date'] >= '2025-01-01') & (df_stations['calc_date'] <= '2025-12-31')].copy()
df_stations = df_stations.dropna(subset=['daily_avg_pm25', 'site_id', 'calc_date'])
df_stations = df_stations[df_stations['hours_count'] >= 12] 

df_stations = df_stations.sort_values(by=['site_id', 'calc_date', 'hours_count'], ascending=[True, True, False])
df_stations = df_stations.drop_duplicates(subset=['site_id', 'calc_date'], keep='first').reset_index(drop=True)
print(f"🧹 Data Hardening: Retained {len(df_stations)} verified primary monitor observations.")

# --- Load Stage 1 Model ---
booster_stg1 = xgb.Booster()
booster_stg1.load_model(STAGE1_MODEL_PATH)

metadata_cols = [
    'device_name', 'calc_date', 'daily_avg_pm25', 'hours_count', 'latitude', 'longitude', 
    'site_id', 'site_name', 'lat', 'lon', 'weight', 'month', 'geom', 'sub_county',
    'pm25_stage1_pred_log', 'pm25_stage1_pred_log_OOF', 'pm25_stage1_pred_physical_OOF', 
    'pm25_stage1_pred_physical', 'DC', 'improved_Ps'
]
features = [c for c in df_stations.columns if c not in metadata_cols]

# =====================================================================
# 2. MASTER GRID CONSTRUCTION
# =====================================================================
print("\n[1/4] Constructing Dense Skeleton Grid and Sampling Static Covariates...")

with rasterio.open(STATIC_RASTERS['elevation']) as src:
    width, height = src.width, src.height
    transform = src.transform
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    u_x, u_y = rasterio.transform.xy(transform, rows, cols, offset='center')
    
    data = src.read(1).flatten()
    nodata_val = src.nodata
    valid_mask = ~np.isnan(data) if (nodata_val is None or np.isnan(nodata_val)) else (data != nodata_val) & ~np.isnan(data)

master_pixels = pd.DataFrame({
    'utm_x': np.array(u_x).flatten()[valid_mask], 
    'utm_y': np.array(u_y).flatten()[valid_mask]
}).reset_index(drop=True)

gdf_pixels = gpd.GeoDataFrame(
    master_pixels, 
    geometry=gpd.points_from_xy(master_pixels.utm_x, master_pixels.utm_y), 
    crs="EPSG:32737"
).to_crs("EPSG:4326")

master_pixels['lon'] = gdf_pixels.geometry.x.values
master_pixels['lat'] = gdf_pixels.geometry.y.values

static_pixel_coords = list(zip(master_pixels['utm_x'], master_pixels['utm_y']))
for var, path in STATIC_RASTERS.items():
    with rasterio.open(path) as src:
        samples = [v[0] for v in list(src.sample(static_pixel_coords))]
        master_pixels[var] = np.nan_to_num(samples, nan=0.0) if var in PHYSICAL_ZERO_FEATS else samples

# =====================================================================
# 3. DAILY EXTRACTION & LEAK-FREE NEIGHBORHOOD CANDIDATE PRE-CALCULATION
# =====================================================================
print("\n[2/4] Simulating Daily Extents, Running Stage 1, and Indexing Isolated Spatial Pools...")

unique_dates = sorted(df_stations['calc_date'].unique())
EARTH_RADIUS_KM = 6371.0

# Arrays corresponding directly to rows in df_stations to hold neighbor metrics (up to max CV k=10)
all_neighbor_dists = [None] * len(df_stations)
all_neighbor_vals = [None] * len(df_stations)

date_groups = df_stations.groupby('calc_date').groups

for date in unique_dates:
    if date not in date_groups: continue
    row_idxs = date_groups[date]
    df_day_obs = df_stations.loc[row_idxs]
    
    dt = pd.to_datetime(date)
    doy, month_str, date_str = dt.dayofyear, dt.strftime('%m'), dt.strftime('%Y-%m-%d')
    
    df_grid = master_pixels.copy()
    total_grid_size = len(df_grid)
    
    # Exclude grid locations containing active ground monitors today
    tree_grid = BallTree(np.radians(df_grid[['lat', 'lon']].values), metric='haversine')
    st_coords = np.radians(df_day_obs[['lat', 'lon']].values)
    _, occupied_idxs = tree_grid.query(st_coords, k=1)
    
    unique_occupied_idxs = list(set(occupied_idxs.flatten()))
    unmonitored_idxs = df_grid.index.difference(unique_occupied_idxs)
    
    # Temporal Encodings
    df_grid['calc_date'], df_grid['month'], df_grid['day_of_week'] = date, dt.month, dt.dayofweek
    df_grid['t1_n'] = np.cos(2 * np.pi * doy / 365 - np.pi / 12)
    df_grid['t2_n'] = np.sin(2 * np.pi * doy / 365 - np.pi / 12)
    df_grid['t3'] = doy / 365
    df_grid['st_interact_1'] = df_grid['t1_n'] * df_grid['lat']
    df_grid['st_interact_2'] = df_grid['t2_n'] * df_grid['lon']
    
    pixel_coords = list(zip(df_grid['utm_x'], df_grid['utm_y']))
    
    # Dynamic Spatial Extractions
    ndvi_path = os.path.join(NDVI_DIR, f"aligned_nairobi_ndvi_2025_{month_str}.tif")
    df_grid['ndvi'] = np.nan
    if os.path.exists(ndvi_path):
        with rasterio.open(ndvi_path) as src: df_grid['ndvi'] = [v[0] for v in src.sample(pixel_coords)]
    
    era5_path = os.path.join(ERA5_DIR, f"aligned_Nairobi_Met_{date_str}.tif")
    for b in ERA5_BANDS: df_grid[b] = np.nan
    if os.path.exists(era5_path):
        with rasterio.open(era5_path) as src:
            samples = list(src.sample(pixel_coords, indexes=list(range(1, 8))))
            for i, b in enumerate(ERA5_BANDS): df_grid[b] = [s[i] for s in samples]

    # Meteorological Lags
    dt_m1, dt_m2 = dt - pd.Timedelta(days=1), dt - pd.Timedelta(days=2)
    era5_m1_path = os.path.join(ERA5_DIR, f"aligned_Nairobi_Met_{dt_m1.strftime('%Y-%m-%d')}.tif")
    era5_m2_path = os.path.join(ERA5_DIR, f"aligned_Nairobi_Met_{dt_m2.strftime('%Y-%m-%d')}.tif")
    df_grid['PBL_Height_lag1'], df_grid['Relative_Humidity_lag1'] = np.nan, np.nan
    precip_m1, precip_m2 = np.zeros(total_grid_size), np.zeros(total_grid_size)

    if os.path.exists(era5_m1_path):
        with rasterio.open(era5_m1_path) as src:
            samples_m1 = list(src.sample(pixel_coords, indexes=[1, 6, 7]))
            df_grid['Relative_Humidity_lag1'] = [s[0] for s in samples_m1]
            df_grid['PBL_Height_lag1'] = [s[1] for s in samples_m1]
            precip_m1 = np.array([s[2] for s in samples_m1])

    if os.path.exists(era5_m2_path):
        with rasterio.open(era5_m2_path) as src:
            samples_m2 = list(src.sample(pixel_coords, indexes=[7]))
            precip_m2 = np.array([s[0] for s in samples_m2])
            
    df_grid['Precip_3d_sum'] = df_grid['Total_Precipitation_mm'].fillna(0).values + precip_m1 + precip_m2

    v_search = glob.glob(os.path.join(VIIRS_DIR, f"clipped_VIIRS_Nairobi_UTM_A2025{dt.strftime('%j')}*.tif"))
    for b in VIIRS_BANDS: df_grid[b] = np.nan
    if v_search:
        with rasterio.open(v_search[0]) as src:
            samples = list(src.sample(pixel_coords, indexes=list(range(1, 8))))
            for i, b in enumerate(VIIRS_BANDS): df_grid[b] = [s[i] for s in samples]

    for var, fname in {'NO2': 'NO2_umol_m2.tif', 'CO': 'CO_mmol_m2.tif', 'SO2': 'SO2_umol_m2.tif'}.items():
        g_path = os.path.join(GASES_DIR, fname)
        df_grid[var] = np.nan
        if os.path.exists(g_path):
            with rasterio.open(g_path) as src: df_grid[var] = [v[0] for v in src.sample(pixel_coords, indexes=[doy])]

    df_grid['Wind_Speed_10m'] = np.sqrt(df_grid['U_Wind_10m']**2 + df_grid['V_Wind_10m']**2)
    for col in ['viirs_SAA', 'viirs_VAA']:
        if col in df_grid.columns:
            df_grid[f'{col}_sin'], df_grid[f'{col}_cos'] = np.sin(np.radians(df_grid[col])), np.cos(np.radians(df_grid[col]))
    
    # --- MAINTAINED: Generate Supplementary Pool via Stage 1 Model ---
    df_unmonitored = df_grid.loc[unmonitored_idxs].copy()
    for f in features:
        if f not in df_unmonitored.columns: df_unmonitored[f] = np.nan
            
    dmatrix_input = xgb.DMatrix(df_unmonitored[features].astype('float32'))
    df_unmonitored['predicted_pm25_physical'] = booster_stg1.predict(dmatrix_input)
    
    inspection_file = os.path.join(DAILY_INSPECTION_DIR, f"nairobi_predictions_{date_str}.csv")
    df_unmonitored[['lat', 'lon', 'predicted_pm25_physical']].to_csv(inspection_file, index=False)

    # --- CRITICAL FIX: Build Structurally Isolated Spatial Pool (No Sensor Leakage) ---
    df_combined_pool = df_unmonitored[['lat', 'lon', 'predicted_pm25_physical']].rename(columns={'predicted_pm25_physical': 'pm25_val'})
    
    combined_vals = df_combined_pool['pm25_val'].values
    combined_coords_rad = np.radians(df_combined_pool[['lat', 'lon']].values)
    
    # Query up to 10 neighbors to satisfy the logging optimization loop
    tree_pool = BallTree(combined_coords_rad, metric='haversine')
    distances_list, indices_list = tree_pool.query(st_coords, k=10)
    
    # Map neighbor metrics back globally (monitors are blind to other monitors)
    for idx_in_day, global_idx in enumerate(row_idxs):
        all_neighbor_dists[global_idx] = distances_list[idx_in_day] * EARTH_RADIUS_KM
        all_neighbor_vals[global_idx] = combined_vals[indices_list[idx_in_day]]

# =====================================================================
# 4. DOWNSTREAM STAGE 2 OPTIMIZATION VIA 5-FOLD CV & METRICS
# =====================================================================
print("\n[3/4] Optimizing k Based on Downstream Stage 2 XGBoost Performance...")

X_base = df_stations[features].fillna(0).values.astype('float32')
y_true = df_stations['daily_avg_pm25'].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary to hold computed vectors for later assignment
ps_store = {}

for k in range(1, 11):
    ps_column = []
    
    for i in range(len(df_stations)):
        sub_dist = all_neighbor_dists[i][:k]
        sub_pm25 = all_neighbor_vals[i][:k]
        
        if len(sub_dist) == 0:
            ps_column.append(0.0)
            continue
            
        # Local Adaptive Bandwidth Strategy
        h_i = sub_dist[-1] if sub_dist[-1] > 0 else 0.001
        weights = np.exp(-(sub_dist**2) / (h_i**2))
        w_sum = np.sum(weights)
        
        ps_val = np.sum(weights * sub_pm25) / w_sum if w_sum > 0 else np.mean(sub_pm25)
        ps_column.append(ps_val)
        
    ps_column = np.array(ps_column)
    ps_store[k] = ps_column
    
    # Append dynamic feature column to matrix
    X_stage2 = np.column_stack((X_base, ps_column))
    
    fold_rmses = []
    fold_pearsons = []
    fold_spearmans = []
    
    for train_idx, val_idx in kf.split(X_stage2):
        X_train, X_val = X_stage2[train_idx], X_stage2[val_idx]
        y_train, y_val = y_true[train_idx], y_true[val_idx]
        
        model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        fold_rmses.append(root_mean_squared_error(y_val, preds))
        
        # Pearson Correlation
        r_p, _ = pearsonr(y_val, preds)
        fold_pearsons.append(r_p)
        
        # Spearman Rank Correlation
        rho, _ = spearmanr(y_val, preds)
        fold_spearmans.append(rho)
        
    mean_rmse = np.mean(fold_rmses)
    mean_pearson = np.mean(fold_pearsons)
    mean_spearman = np.mean(fold_spearmans)
    print(f"Testing k = {k:2d} | Stage 2 CV RMSE = {mean_rmse:.4f} | Pearson 𝑟 = {mean_pearson:.4f} | Spearman 𝜌 = {mean_spearman:.4f}")

print(f"\n🎯 CV Loop Logging Complete! Enforcing chosen scientific neighborhood size k = 3...")

# =====================================================================
# 5. ASSIGNMENT & FINAL OUTPUT EXPORT
# =====================================================================
print("\n[4/4] Structuring Dataset and Exporting CSV...")

# Hardcoding the feature extraction selection to k = 3 as requested
best_k = 3
df_stations['improved_Ps'] = ps_store[best_k]

# Handle any anomalous remaining missing cells by grouping temporally 
df_stations['improved_Ps'] = df_stations.groupby('calc_date')['improved_Ps'].transform(lambda x: x.fillna(x.mean()))
df_stations['improved_Ps'] = df_stations['improved_Ps'].fillna(df_stations['daily_avg_pm25'].mean())

if 'pm25_stage1_pred_physical' in df_stations.columns:
    df_stations = df_stations.drop(columns=['pm25_stage1_pred_physical'])

df_stations.to_csv(OUTPUT_STAGE2_CSV, index=False)
print(f"EXPORT SUCCESSFUL: Balanced leak-free dataset written out to {OUTPUT_STAGE2_CSV}")