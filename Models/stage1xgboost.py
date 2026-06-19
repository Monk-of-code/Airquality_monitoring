import time
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import halfnorm
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# -------------------------------------------------------
# 1. LOAD DATA & FILTER STRICTLY FOR 2025
# -------------------------------------------------------
print("Loading Nairobi dataset Adaptation...")
df = pd.read_csv('/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_training_dataset.csv')

# Ensure valid datetime parsing and restrict strictly to the 2025 period
df['calc_date'] = pd.to_datetime(df['calc_date'])
df = df[(df['calc_date'] >= '2025-01-01') & (df['calc_date'] <= '2025-12-31')].copy()

# Drop rows missing critical identifiers or target values
df = df.dropna(subset=['daily_avg_pm25', 'hours_count', 'site_id', 'device_name'])

# Clip hours_count safely to a maximum baseline of 24
df['hours_count'] = df['hours_count'].clip(upper=24)
initial_n = len(df)

# METHODOLOGY UPDATE: Enforcing a 12-hour coverage rule to maintain spatial representation
df = df[df['hours_count'] >= 16].reset_index(drop=True)
print(f"Applied 16hr coverage rule: {len(df)} days remaining (Filtered {initial_n - len(df)} noisy records).")

# -------------------------------------------------------
# 2. PRIMARY DEVICE SELECTION (ZERO LOCATION LEAKAGE)
# -------------------------------------------------------
print("\nEvaluating collocated devices per site...")

# Compute average hour counts and total recorded days for each individual device
device_stats = df.groupby(['site_id', 'device_name']).agg(
    avg_hours=('hours_count', 'mean'),
    days_recorded=('calc_date', 'count')
).reset_index()

# Sort devices within each site: highest average hours first, then most days recorded
device_stats = device_stats.sort_values(
    by=['site_id', 'avg_hours', 'days_recorded'], 
    ascending=[True, False, False]
)

# Isolate the single top-ranked primary device for each site
primary_devices = device_stats.groupby('site_id').first().reset_index()
print(f"Identified {len(primary_devices)} unique primary monitoring sites.")

# Discard all rows belonging to secondary/redundant devices to prevent feature mixing
df = df.merge(primary_devices[['site_id', 'device_name']], on=['site_id', 'device_name'], how='inner')
print(f"Dataset filtered successfully. Retained {len(df)} primary physical observations.")

# -------------------------------------------------------
# 3. FORMAL HALF-NORMAL WEIGHTING & CHEN ET AL. ENCODING
# -------------------------------------------------------
# Implement weights using scipy's halfnorm distribution
# Centered at 24 hours (deviation x = 24 - hours_count) with scale (sigma) = 8
deviations = (24.0 - df['hours_count']).values
formal_weights = halfnorm.pdf(deviations, loc=0, scale=8)

# Normalize weights so a perfect 24-hour record equals exactly 1.0 weight
max_pdf_val = halfnorm.pdf(0, loc=0, scale=8)
df['weight'] = (formal_weights / max_pdf_val).astype('float32')

# Adjusted counterclockwise by pi/12 for seasonal symmetry over a fixed 365-day year
day_of_year = df['calc_date'].dt.dayofyear
df['t1_n'] = np.cos(2 * np.pi * day_of_year / 365.0 - np.pi / 12)
df['t2_n'] = np.sin(2 * np.pi * day_of_year / 365.0 - np.pi / 12)
df['t3']   = day_of_year / 365.0

# Spatiotemporal Interaction Features
df['st_interact_1'] = df['t1_n'] * df['lat']
df['st_interact_2'] = df['t2_n'] * df['lon']

# -------------------------------------------------------
# 4. COMPREHENSIVE FEATURE SELECTION
# -------------------------------------------------------
# Dynamically separate metadata targets from active training features
# Standardized spatial coordinates to 'lat' and 'lon' exclusively
metadata_cols = [
    'device_name', 'calc_date', 'daily_avg_pm25', 'hours_count', 
    'site_id', 'site_name', 'lat', 'lon', 'weight', 'month','geom','sub_county'
]

features = [c for c in df.columns if c not in metadata_cols]
print(f"\nFeatures included in Stage 1 ({len(features)} total): {features}")

X = df[features].astype('float32')

# METHODOLOGY UPDATE: Directly predicting raw physical values. No Log Transformation.
y_target = df['daily_avg_pm25'].values.astype('float32')
weights_array = df['weight'].values.astype('float32')

groups_site = df['site_id'].values
groups_time = df['calc_date'].values

# -------------------------------------------------------
# 5. HYPERPARAMETER TUNING (GPU-Accelerated Gamma Objective)
# -------------------------------------------------------
print("\nInitiating Spatially-Isolated Randomized Search...")
tuning_model = xgb.XGBRegressor(
    tree_method='hist', 
    device='cuda', 
    objective='reg:gamma',    # Handles right-skewed PM2.5 natively without dampening peaks
    eval_metric='mae',        # Optimizing directly for absolute error
    random_state=42
)

param_grid = {
    'n_estimators': [1500, 2000, 3000],
    'learning_rate': [0.005, 0.01, 0.02],
    'max_depth': [4, 6, 8],
    'min_child_weight': [20, 30, 50],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.5, 0.6, 0.7],
    'gamma': [0.5, 1.0, 3.0],
    'alpha': [0.1, 1.0, 3.0], 
    'reg_lambda': [10.0, 25.0, 50.0]
}

search = RandomizedSearchCV(
    tuning_model, 
    param_grid, 
    n_iter=150, 
    cv=GroupKFold(n_splits=5), 
    scoring='neg_mean_absolute_error', # Aligning scoring with Gamma objective 
    verbose=2, 
    random_state=42,
    n_jobs=1
)

search.fit(X, y_target, groups=groups_site, sample_weight=weights_array)
best_params = search.best_params_
print(f"\nBest Parameters Found: {best_params}")

# -------------------------------------------------------
# 6. TRIPLE ACCURACY VALIDATION & OOF EXTRACTION
# -------------------------------------------------------
print("\nPreparing C-contiguous memory buffers to bypass CUDA slicing exceptions...")

X_arr = np.ascontiguousarray(X.values, dtype=np.float32)
y_target_arr = np.ascontiguousarray(y_target, dtype=np.float32)
weights_arr = np.ascontiguousarray(weights_array, dtype=np.float32)

print("Executing Spatiotemporal Cross-Validation Loops...")

def run_cv_standard(splitter, groups=None):
    pred_oof = np.zeros_like(y_target_arr)
    eval_params = {**best_params, 'tree_method': 'hist', 'device': 'cuda', 'objective': 'reg:gamma', 'eval_metric': 'mae', 'random_state': 42}
    
    for tr_idx, te_idx in splitter.split(X_arr, y_target_arr, groups):
        fold_model = xgb.XGBRegressor(**eval_params, n_jobs=1)
        
        X_train_slice = np.ascontiguousarray(X_arr[tr_idx])
        y_train_slice = np.ascontiguousarray(y_target_arr[tr_idx])
        w_train_slice = np.ascontiguousarray(weights_arr[tr_idx])
        X_test_slice  = np.ascontiguousarray(X_arr[te_idx])
        
        fold_model.fit(X_train_slice, y_train_slice, sample_weight=w_train_slice)
        pred_oof[te_idx] = fold_model.predict(X_test_slice)
    
    # Direct physical space evaluation
    r2 = r2_score(y_target_arr, pred_oof)
    mae = mean_absolute_error(y_target_arr, pred_oof)
    rmse = np.sqrt(mean_squared_error(y_target_arr, pred_oof))
    mre = np.mean(np.abs((pred_oof - y_target_arr) / y_target_arr)) * 100
    
    return [r2, mae, rmse, mre], pred_oof

val_sample, _ = run_cv_standard(KFold(n_splits=10, shuffle=True, random_state=42))

# Extract spatial OOF predictions to prevent spatial target leakage in Stage 2.
val_site, site_oof_preds = run_cv_standard(GroupKFold(n_splits=10), groups=groups_site)
val_time, _ = run_cv_standard(GroupKFold(n_splits=10), groups=groups_time)

# -------------------------------------------------------
# 7. FINAL RESULTS OUTPUT & EXPORT
# -------------------------------------------------------
results = pd.DataFrame([val_sample, val_site, val_time], 
                       index=['Sample-based CV', 'Site-based CV', 'Time-based CV'],
                       columns=['R2', 'MAE', 'RMSE', 'MRE (%)'])

print("\n" + "="*70)
print(" STAGE 1 VALIDATION REPORT (Chen & Kloog Enhanced)")
print("="*70)
print(results.round(4))
print("="*70)

# Save the OOF predictions (already in physical space) to the dataset for Stage 2
df['pm25_stage1_pred_physical_OOF'] = site_oof_preds

print("\nFitting final production model on full dataset to lock feature schemas...")
final_production_model = xgb.XGBRegressor(**best_params, tree_method='hist', device='cuda', objective='reg:gamma', eval_metric='mae', random_state=42)
final_production_model.fit(X, y_target, sample_weight=weights_array)

final_production_model.save_model('/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stg1_xgboost_v3.json')

# Retain full-fit predictions purely for reference/inference
df['pm25_stage1_pred_physical'] = final_production_model.predict(X)
df.to_csv('/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stg1_results_v3.csv', index=False)

print(f" Stage 1 complete. Dataset uniquely prepared for rigorous Stage 2 residuals.")