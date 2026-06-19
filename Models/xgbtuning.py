
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error
import time
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------
# 1. LOAD DATA & CALCULATE WEIGHTS
# -------------------------------------------------------
print("Loading CLEANED dataset for Empirical XGBoost Tuning...")
df = pd.read_csv('/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_ml_cleaned_training_dataset.csv')

# Cap hours_count to handle any imputation anomalies
df['hours_count'] = df['hours_count'].clip(upper=24)

print("Calculating Half-Normal sample weights...")
sigma = 10
weights = np.exp(-((24 - df['hours_count'])**2) / (2 * sigma**2))

# -------------------------------------------------------
# 2. CYCLIC TIME TRANSFORMATION
# -------------------------------------------------------
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# -------------------------------------------------------
# 3. FEATURE SELECTION (TEMP_2M INCLUDED)
# -------------------------------------------------------
final_columns = [
    # Cyclic Temporal
    'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
    # Static Spatial
    'industry_influence', 'transit_influence', 'pop_density', 
    'road_density', 'building_heights', 'elevation',
    # Satellites & Geometry
    'ndvi', 'viirs_M03', 'viirs_M11', 'viirs_SZA', 'viirs_VZA',               
    'viirs_SAA_sin', 'viirs_SAA_cos', 'viirs_VAA_sin', 'viirs_VAA_cos',       
    # Trace Gases 
    'NO2', 'CO', 'SO2',
    # Meteorology / Atmosphere
    'Total_Precipitation_mm', 'U_Wind_10m', 'V_Wind_10m',             
    'Relative_Humidity', 'PBL_Height', 
    'Temp_2m'  # Let the model decide!
]

X = df[final_columns]
y = np.log1p(df['daily_avg_pm25'])      # log transform target

groups = df['site_id']

print(f"Training on {len(X)} rows across {groups.nunique()} unique physical locations.")
print(f"Total Features: {len(final_columns)}")

# -------------------------------------------------------
# 4. CONFIGURE THE RTX 5070 GPU
# -------------------------------------------------------
try:
    xgb_model = xgb.XGBRegressor(tree_method='hist', device='cuda',
                                 objective='reg:squarederror', random_state=42)
except Exception as e:
    print("CUDA unavailable, falling back to CPU...")
    xgb_model = xgb.XGBRegressor(tree_method='hist', objective='reg:squarederror', random_state=42)

# L1 Regularization (alpha) to force the model to drop useless features
param_distributions = {
    'n_estimators': [1000, 1500, 2000, 3000],
    'learning_rate': [0.005, 0.01, 0.02, 0.05],
    'max_depth': [4, 6, 8, 10],                       
    'min_child_weight': [10, 20, 30,50],              
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
    'gamma': [0, 0.1, 0.5, 1, 3],
    'alpha': [0, 0.1, 0.5, 1, 3],            
    'reg_lambda': [1, 5, 10, 25, 50]
}

# -------------------------------------------------------
# 5. EXECUTE TUNING (Spatially‑Isolated)
# -------------------------------------------------------
gkf = GroupKFold(n_splits=10)

search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=100,
    scoring=['r2', 'neg_root_mean_squared_error'],
    refit='r2',
    cv=gkf,
    verbose=2,
    random_state=42,
    n_jobs=1
)

print("\n Initiating SPATIALLY‑ISOLATED GPU Search...")
start_time = time.time()

# Convert to float32 for GPU efficiency
X = X.astype('float32')
y = y.astype('float32')
weights = weights.astype('float32')

# Fit the random search (sample_weight is sliced correctly by RandomizedSearchCV internally)
search.fit(X, y, groups=groups, **{'sample_weight': weights})
elapsed_time = (time.time() - start_time) / 60

# -------------------------------------------------------
# 6. RESULTS & TRUE OUT‑OF‑SAMPLE RMSE
# -------------------------------------------------------
print(f"\n Tuning Complete in {elapsed_time:.2f} minutes!")
print("-" * 50)
print(" BEST PARAMETERS FOUND:")
for key, value in search.best_params_.items():
    print(f"   {key}: {value}")

print("-" * 50)
best_r2 = search.cv_results_['mean_test_r2'][search.best_index_]
print(f" CV R‑SQUARED (Log Space): {best_r2:.4f}")


print("\nCalculating TRUE Out‑of‑Sample Physical RMSE (manual, weighted)...")
pred_log_oof = np.zeros_like(y, dtype=np.float32)

for train_idx, test_idx in gkf.split(X, y, groups):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    w_tr = weights[train_idx]             # slice weights to match training data

    # Clone the best estimator using discovered parameters
    fold_model = xgb.XGBRegressor(**search.best_params_,
                                  tree_method='hist', device='cuda',
                                  objective='reg:squarederror', random_state=42)
    fold_model.fit(X_tr, y_tr, sample_weight=w_tr)
    pred_log_oof[test_idx] = fold_model.predict(X_te)

# Inverse transform back to physical µg/m³
true_physical = np.expm1(y)
pred_physical_oof = np.expm1(pred_log_oof)
physical_rmse_oof = np.sqrt(mean_squared_error(true_physical, pred_physical_oof))

print(f"  GENERALIZATION RMSE: {physical_rmse_oof:.4f} µg/m³")

# -------------------------------------------------------
# 7. SAVE THE FINAL MODEL
# -------------------------------------------------------
best_model = search.best_estimator_   # the model refit on all data with best params
best_model.save_model('/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_xgboost_v2.json')
print("\n Model saved! (Outputs in LOG SPACE. Use np.expm1() to map to µg/m³)")