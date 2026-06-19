import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# =====================================================================
# 1. SETUP & CLEANING
# =====================================================================
DATA_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stage2_training.csv'
MODEL_SAVE_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stg2_rf_model.joblib'

print("Loading completed Stage 2 dataset...")
df = pd.read_csv(DATA_PATH)
df['calc_date'] = pd.to_datetime(df['calc_date'])

# Drop redundant collocated devices to prevent spatial CV leakage
initial_rows = len(df)
df = df.sort_values(by=['site_id', 'calc_date', 'hours_count'], ascending=[True, True, False])
df = df.drop_duplicates(subset=['site_id', 'calc_date'], keep='first').reset_index(drop=True)
print(f"Dropped {initial_rows - len(df)} redundant collocated device rows.")

weights_array = df['weight'].values.astype('float32')

metadata_cols = [
    'device_name', 'calc_date', 'daily_avg_pm25', 'hours_count', 
    'site_id', 'site_name', 'lat', 'lon', 'month', 'weight', 'geom', 'sub_county',
    'pm25_stage1_pred_log', 'pm25_stage1_pred_log_OOF', 'pm25_stage1_pred_physical_OOF'
]
original_features = [c for c in df.columns if c not in metadata_cols]

y_phys = df['daily_avg_pm25'].values.astype('float32')
y_log = np.log1p(y_phys)

groups_site = df['site_id'].values
groups_time = df['calc_date'].values

# =====================================================================
# 2. ALGORITHMIC FEATURE PRUNING (VIA RANDOM FOREST)
# =====================================================================
print(f"\nEvaluating baseline importance of all {len(original_features)} features...")
X_full = df[original_features].astype('float32')

# Fast baseline RF for feature importance
pruner_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
pruner_model.fit(X_full, y_log, sample_weight=weights_array)

importance_df = pd.DataFrame({
    'Feature': original_features,
    'Importance': pruner_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

top_features = importance_df.head(38)['Feature'].tolist()

if 'improved_Ps' not in top_features:
    top_features.append('improved_Ps')

print("\n--- SURVIVING FEATURES (TOP TIER) ---")
for i, f in enumerate(top_features):
    print(f"{i+1}. {f} ({importance_df[importance_df['Feature']==f]['Importance'].values[0]:.4f})")

print(f"\nSlicing dataset: Dropping {len(original_features) - len(top_features)} weak/noisy features.")
X = df[top_features].astype('float32')

# =====================================================================
# 3. UNBIASED RANDOMIZED SEARCH CV
# =====================================================================
print("\nInitiating Unbiased Parallel Randomized Search on Pruned Dataset...")

tuning_model = RandomForestRegressor(random_state=42)

# Random Forest requires deeper trees than XGBoost, but heavy leaf restrictions to prevent memorization
param_grid = {
    'n_estimators': [500, 1000, 1500],
    'max_depth': [10, 15, 20, None],          # Allow deeper growth
    'min_samples_split': [5, 10, 20],         
    'min_samples_leaf': [4, 8, 12],           # Force terminal nodes to contain multiple samples (smooths noise)
    'max_features': ['sqrt', 'log2', 0.5]     # CRITICAL: Prevents improved_Ps from dominating the forest
}

search = RandomizedSearchCV(
    tuning_model, 
    param_grid, 
    n_iter=50,                              # Fewer iterations needed for RF compared to XGBoost
    cv=GroupKFold(n_splits=5),              
    scoring='neg_mean_squared_error', 
    verbose=2, 
    random_state=42,
    n_jobs=-1                               # Use all CPU cores for fast parallel tree building
)

search.fit(X, y_log, groups=groups_site, sample_weight=weights_array)
best_params = search.best_params_
print(f"\nOptimal Stage 2 RF Parameters Found: {best_params}")

# =====================================================================
# 4. TRIPLE CROSS-VALIDATION EVALUATION
# =====================================================================
X_arr = np.ascontiguousarray(X.values, dtype=np.float32)
y_log_arr = np.ascontiguousarray(y_log, dtype=np.float32)
weights_arr = np.ascontiguousarray(weights_array, dtype=np.float32)

print("\nExecuting Final Spatiotemporal Cross-Validation Loops...")

def run_triple_cv(splitter, groups=None):
    pred_log_oof = np.zeros_like(y_log_arr)
    eval_params = {**best_params, 'random_state': 42, 'n_jobs': -1}
    
    for tr_idx, te_idx in splitter.split(X_arr, y_log_arr, groups):
        fold_model = RandomForestRegressor(**eval_params)
        
        X_train_slice = np.ascontiguousarray(X_arr[tr_idx])
        y_train_slice = np.ascontiguousarray(y_log_arr[tr_idx])
        w_train_slice = np.ascontiguousarray(weights_arr[tr_idx])
        X_test_slice  = np.ascontiguousarray(X_arr[te_idx])
        
        fold_model.fit(X_train_slice, y_train_slice, sample_weight=w_train_slice)
        pred_log_oof[te_idx] = fold_model.predict(X_test_slice)
        
    pred_phys = np.expm1(pred_log_oof)
    r2 = r2_score(y_phys, pred_phys)
    mae = mean_absolute_error(y_phys, pred_phys)
    rmse = np.sqrt(mean_squared_error(y_phys, pred_phys))
    mre = np.mean(np.abs((pred_phys - y_phys) / y_phys)) * 100
    return [r2, mae, rmse, mre]

val_sample = run_triple_cv(KFold(n_splits=10, shuffle=True, random_state=42))
val_site   = run_triple_cv(GroupKFold(n_splits=10), groups=groups_site)
val_time   = run_triple_cv(GroupKFold(n_splits=10), groups=groups_time)

results = pd.DataFrame([val_sample, val_site, val_time], 
                       index=['Sample-based CV', 'Site-based CV', 'Time-based CV'],
                       columns=['R2', 'MAE', 'RMSE', 'MRE (%)'])

print("\n" + "="*70)
print(" STAGE 2 RANDOM FOREST VALIDATION REPORT")
print("="*70)
print(results.round(4))
print("="*70)

# =====================================================================
# 5. FINAL PRODUCTION FIT & EXPORT
# =====================================================================
print("\nFitting final Stage 2 Random Forest production model...")

final_production_model = RandomForestRegressor(**{**best_params, 'random_state': 42, 'n_jobs': -1})
final_production_model.fit(X, y_log, sample_weight=weights_array)

# Save the model using joblib (standard for scikit-learn)
joblib.dump(final_production_model, MODEL_SAVE_PATH)
print(f"Success! Final Stage 2 Random Forest Model securely exported to: {MODEL_SAVE_PATH}")