import os
import gc
import numpy as np
import pandas as pd
import cupy as cp  # Added for direct GPU memory management
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, GroupKFold, ParameterSampler

# =====================================================================
# 1. CONFIGURATION & DATA LOADING
# =====================================================================
print("Initializing Stage 2 Spatiotemporal Pipeline (VRAM Optimized)...")

INPUT_CSV = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stage2_training.csv'
MODEL_SAVE_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stage2_production_model.json'

# Explicitly defining the GPU environment
HARDWARE_CONFIG = {
    'device': 'cuda',       # Forces XGBoost to expect GPU structures
    'tree_method': 'hist'   # GPU-accelerated histogram binning
}

print(f"Loading prepared Stage 2 dataset from: {INPUT_CSV}")
try:
    df_stage2 = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    raise FileNotFoundError(f"🚨 ERROR: Could not find {INPUT_CSV}. Run Stage 1 first.")

y_target = df_stage2['daily_avg_pm25'].values
weights_array = df_stage2['weight'].values if 'weight' in df_stage2.columns else np.ones(len(y_target))

groups_site = df_stage2['site_id'].values 
groups_time = df_stage2['calc_date'].values 

metadata_cols = [
    'device_name', 'calc_date', 'daily_avg_pm25', 'hours_count', 'latitude', 'longitude', 
    'site_id', 'site_name', 'lat', 'lon', 'weight', 'month', 'geom', 'sub_county',
    'pm25_stage1_pred_log', 'pm25_stage1_pred_log_OOF', 'pm25_stage1_pred_physical_OOF', 
    'pm25_stage1_pred_physical' 
]

X_base = df_stage2.drop(columns=[col for col in metadata_cols if col in df_stage2.columns])

print(f"Loaded {len(df_stage2)} observations.")
print(f"Training features ({len(X_base.columns)}): {list(X_base.columns)}")

# =====================================================================
# 2. DYNAMIC RANDOM SEARCH OPTIMIZATION LOOP
# =====================================================================
print("\nInitiating Hyperparameter Space Exploration for Stage 2...")

param_grid = {
    'n_estimators': [1000, 1500, 2000],
    'learning_rate': [0.01, 0.02, 0.05],
    'max_depth': [4, 5, 6, 8],              
    'min_child_weight': [10, 20, 30, 50],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'gamma': [0.1, 0.5, 1.0],
    'alpha': [0.1, 1.0, 3.0],                
    'reg_lambda': [5.0, 10.0, 20.0]          
}

param_list = list(ParameterSampler(param_grid, n_iter=150, random_state=42)) 
best_score = float('inf')
best_params = None

spatial_splitter = GroupKFold(n_splits=5)

# Keep master arrays on CPU RAM
X_arr = np.ascontiguousarray(X_base.values, dtype=np.float32)
y_arr = np.ascontiguousarray(y_target, dtype=np.float32)
w_arr = np.ascontiguousarray(weights_array, dtype=np.float32)

for i, current_params in enumerate(param_list):
    fold_maes = []
    eval_params = {**current_params, **HARDWARE_CONFIG, 'objective': 'reg:squarederror', 'random_state': 42}
    
    for tr_idx, te_idx in spatial_splitter.split(X_arr, y_arr, groups=groups_site):

        
        # 1. Slice on CPU
        X_tr, y_tr, w_tr = X_arr[tr_idx], y_arr[tr_idx], w_arr[tr_idx]
        X_te, y_te = X_arr[te_idx], y_arr[te_idx]
        
        # 2. Pin specific fold data to GPU VRAM
        X_tr_gpu = cp.array(X_tr)
        y_tr_gpu = cp.array(y_tr)
        w_tr_gpu = cp.array(w_tr)
        X_te_gpu = cp.array(X_te)
        
        # 3. Fit and predict natively on GPU
        model = xgb.XGBRegressor(**eval_params)
        model.fit(X_tr_gpu, y_tr_gpu, sample_weight=w_tr_gpu)
        preds_gpu = model.predict(X_te_gpu)
        
        # 4. Bring predictions back to CPU for sklearn metric evaluation
        preds = cp.asnumpy(preds_gpu)
        fold_maes.append(mean_absolute_error(y_te, preds))
        
        # 5. AGGRESSIVE VRAM CLEANUP (Prevents WSL2 OOM Crash)
        del model, X_tr_gpu, y_tr_gpu, w_tr_gpu, X_te_gpu, preds_gpu
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        
    avg_mae = np.mean(fold_maes)
    
    # Progress tracker
    print(f"[Iter {i+1}/{len(param_list)}] Tested MAE: {avg_mae:.4f} | Current Best: {best_score:.4f}")
    
    if avg_mae < best_score:
        best_score = avg_mae
        best_params = current_params
        print(f"   🌟 *** NEW BEST SPATIAL MAE: {best_score:.4f} *** 🌟")

print(f"\n✅ Optimal Stage 2 Parameters Found: {best_params}")

# =====================================================================
# 3. TRIPLE CROSS-VALIDATION EVALUATION
# =====================================================================
print("\nExecuting Final Triple-Spatiotemporal Cross-Validation Loops (10-Fold)...")

def run_triple_cv(splitter, groups=None):
    pred_oof = np.zeros(len(df_stage2))
    eval_params = {**best_params, **HARDWARE_CONFIG, 'objective': 'reg:squarederror', 'random_state': 42}
    
    for tr_idx, te_idx in splitter.split(X_arr, y_arr, groups):
        # Move slice to GPU
        X_tr_gpu = cp.array(X_arr[tr_idx])
        y_tr_gpu = cp.array(y_arr[tr_idx])
        w_tr_gpu = cp.array(w_arr[tr_idx])
        X_te_gpu = cp.array(X_arr[te_idx])
        
        fold_model = xgb.XGBRegressor(**eval_params)
        fold_model.fit(X_tr_gpu, y_tr_gpu, sample_weight=w_tr_gpu)
        
        # Predict and pull back to CPU
        preds_gpu = fold_model.predict(X_te_gpu)
        pred_oof[te_idx] = cp.asnumpy(preds_gpu)
        
        # Clear VRAM
        del fold_model, X_tr_gpu, y_tr_gpu, w_tr_gpu, X_te_gpu, preds_gpu
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        
    r2 = r2_score(y_arr, pred_oof)
    mae = mean_absolute_error(y_arr, pred_oof)
    rmse = np.sqrt(mean_squared_error(y_arr, pred_oof))
    mre = np.mean(np.abs((pred_oof - y_arr) / np.maximum(y_arr, 1.0))) * 100 
    
    return [r2, mae, rmse, mre]

val_sample = run_triple_cv(KFold(n_splits=10, shuffle=True, random_state=42))
val_site   = run_triple_cv(GroupKFold(n_splits=10), groups=groups_site)
val_time   = run_triple_cv(GroupKFold(n_splits=10), groups=groups_time)

results = pd.DataFrame([val_sample, val_site, val_time], 
                       index=['Sample-based CV', 'Site-based CV', 'Time-based CV'],
                       columns=['R2', 'MAE', 'RMSE', 'MRE (%)'])

print("\n" + "="*70)
print(" STAGE 2 FINAL VALIDATION REPORT")
print("="*70)
print(results.round(4))
print("="*70)

# =====================================================================
# 4. FINAL PRODUCTION FIT & EXPORT
# =====================================================================
print("\nFitting final Stage 2 production model on full dataset...")

final_production_model = xgb.XGBRegressor(
    **{**best_params, **HARDWARE_CONFIG, 'objective': 'reg:squarederror', 'random_state': 42}
)

# For the final fit ONLY, we pass the Pandas DataFrame (X_base). 
# This triggers one final "DMatrix fallback" warning, but it is necessary 
# so that XGBoost embeds your actual column names into the saved JSON file.
final_production_model.fit(X_base, y_target, sample_weight=weights_array)
final_production_model.save_model(MODEL_SAVE_PATH)

print(f"🎉 Success! Final Stage 2 Production Model securely exported to: {MODEL_SAVE_PATH}")