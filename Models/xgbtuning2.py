# ============================================================================
# Nairobi PM2.5 – XGBoost Tuning (Non‑Seasonal, Spatially‑Isolated)
# ============================================================================
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error
import time
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# 1. Load data and compute half‑normal sample weights
# ----------------------------------------------------------------------------
print("Loading cleaned dataset...")
df = pd.read_csv('/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_ml_cleaned_training_dataset.csv')
df['hours_count'] = df['hours_count'].clip(upper=24)

sigma = 10
weights = np.exp(-((24 - df['hours_count'])**2) / (2 * sigma**2))
weights = weights.astype('float32')
print(f"Weight range: {weights.min():.3f} – {weights.max():.3f}")

# ----------------------------------------------------------------------------
# 2. Cyclic time: day‑of‑week only (no month seasonality)
# ----------------------------------------------------------------------------
# You chose to exclude month sin/cos so the model cannot memorise seasonal
# patterns; it must learn from dynamic variables (wind, humidity, etc.).
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# ----------------------------------------------------------------------------
# 3. Feature set (including Temp_2m) + missingness indicators
# ----------------------------------------------------------------------------
base_features = [
    # Weekly activity rhythm
    'day_of_week_sin', 'day_of_week_cos',
    # Urban morphology
    'industry_influence', 'transit_influence', 'pop_density',
    'road_density', 'building_heights', 'elevation',
    # Satellite aerosol & geometry
    'ndvi', 'viirs_M03', 'viirs_M11', 'viirs_SZA', 'viirs_VZA',
    'viirs_SAA_sin', 'viirs_SAA_cos', 'viirs_VAA_sin', 'viirs_VAA_cos',
    # Trace gases (many missing)
    'NO2', 'CO', 'SO2',
    # Meteorology
    'Total_Precipitation_mm', 'U_Wind_10m', 'V_Wind_10m',
    'Relative_Humidity', 'PBL_Height', 'Temp_2m'
]

# VIIRS M03 and M11 are ~68% missing; adding binary flags allows the model
# to exploit the mere presence/absence of a satellite observation.
df['M03_avail'] = df['viirs_M03'].notna().astype(int)
df['M11_avail'] = df['viirs_M11'].notna().astype(int)

final_columns = base_features + ['M03_avail', 'M11_avail']

X = df[final_columns]                     # pandas DataFrame
y = np.log1p(df['daily_avg_pm25'])        # pandas Series
groups = df['site_id']

print(f"Training: {len(X)} rows, {groups.nunique()} sites, {len(final_columns)} features")
print(f"Target log‑PM2.5 skewness: {y.skew():.2f}")

# ----------------------------------------------------------------------------
# 4. GPU setup
# ----------------------------------------------------------------------------
try:
    _ = xgb.XGBRegressor(tree_method='hist', device='cuda')
    device = 'cuda'
    print("CUDA GPU available.")
except Exception:
    device = 'cpu'
    print("CUDA unavailable – falling back to CPU.")

xgb_model = xgb.XGBRegressor(
    tree_method='hist', device=device,
    objective='reg:squarederror', random_state=42
)

# ----------------------------------------------------------------------------
# 5. Refined hyperparameter grid (Slow Burn)
# ----------------------------------------------------------------------------
param_distributions = {
    'n_estimators': [1000, 1500, 2000, 3000],
    'learning_rate': [0.005, 0.01, 0.02, 0.05],
    'max_depth': [4, 6, 8, 10],
    'min_child_weight': [10, 20, 30, 50],       # consider adding 100 if sites are tiny
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
    'gamma': [0, 0.1, 0.5, 1, 3],
    'alpha': [0, 0.1, 0.5, 1, 3],
    'reg_lambda': [1, 5, 10, 25, 50]
}

# ----------------------------------------------------------------------------
# 6. Spatially‑isolated tuning (GroupKFold = 5 splits)
# ----------------------------------------------------------------------------
gkf = GroupKFold(n_splits=10)   # safer with 75 sites; some have as few as 9 rows

search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=30,
    scoring=['r2', 'neg_root_mean_squared_error'],
    refit='r2',
    cv=gkf,
    verbose=2,
    random_state=42,
    n_jobs=1                     # 1 for GPU; increase if using CPU
)

print("\nStarting non‑seasonal GPU search...")
t0 = time.time()

# XGBoost accepts DataFrames/Series; sample_weight is automatically sliced
# by the CV iterator.
search.fit(X, y, groups=groups, sample_weight=weights)

elapsed = (time.time() - t0) / 60
print(f"\nTuning complete in {elapsed:.1f} minutes")

# ----------------------------------------------------------------------------
# 7. Results & true out‑of‑sample RMSE
# ----------------------------------------------------------------------------
print("\nBest parameters:")
for k, v in search.best_params_.items():
    print(f"   {k}: {v}")

best_r2 = search.cv_results_['mean_test_r2'][search.best_index_]
print(f"\nCV R² (log space): {best_r2:.4f}")

print("\nCalculating out‑of‑sample physical RMSE (manual, weighted)...")
pred_log_oof = np.zeros_like(y, dtype=np.float32)

# Manual loop so we can correctly pass fold‑specific sample weights.
for train_idx, test_idx in gkf.split(X, y, groups):
    # Slice the DataFrame/Series with .iloc, then convert to float32.
    X_tr = X.iloc[train_idx].astype('float32')
    X_te = X.iloc[test_idx].astype('float32')
    y_tr = y.iloc[train_idx].astype('float32')
    w_tr = weights[train_idx]                    # numpy slice

    fold_model = xgb.XGBRegressor(
        **search.best_params_,
        tree_method='hist', device=device,
        objective='reg:squarederror', random_state=42
    )
    fold_model.fit(X_tr, y_tr, sample_weight=w_tr)
    pred_log_oof[test_idx] = fold_model.predict(X_te)

# Back‑transform to µg/m³
true_physical = np.expm1(y)
pred_physical = np.expm1(pred_log_oof)
rmse_phys = np.sqrt(mean_squared_error(true_physical, pred_physical))
print(f"Genuine generalisation RMSE: {rmse_phys:.2f} µg/m³")

# ----------------------------------------------------------------------------
# 8. Feature importance from the final (refitted on all data) model
# ----------------------------------------------------------------------------
best_model = search.best_estimator_
booster = best_model.get_booster()
importance = pd.DataFrame({
    'feature': booster.feature_names,
    'gain': [booster.get_score(importance_type='gain').get(f, 0)
             for f in booster.feature_names]
}).sort_values('gain', ascending=False)

print("\nFeature importance (gain):")
print(importance)

# ----------------------------------------------------------------------------
# 9. Save model and diagnostics
# ----------------------------------------------------------------------------
model_path = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_xgboost_v2_no_month.json'
best_model.save_model(model_path)
importance.to_csv('/mnt/e/Airquality/Datasets/Processed/pm2.5/feature_importance_no_month.csv', index=False)

print(f"\nModel saved to {model_path}")
print("Reminder: outputs are in LOG space. Use np.expm1(pred) for µg/m³.")