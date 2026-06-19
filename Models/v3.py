import warnings
import numpy as np
import pandas as pd
import optuna
import joblib
from scipy.stats import halfnorm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# 1. SETUP, CLEANING & DYNAMIC WEIGHTING
# =====================================================================
DATA_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stage2_training.csv'
MODEL_SAVE_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stg2_final_rf_model.joblib'

print("Loading completed Stage 2 dataset...")
df = pd.read_csv(DATA_PATH)
df['calc_date'] = pd.to_datetime(df['calc_date'])

# Ensure strict temporal ordering for TimeSeriesSplit later
df = df.sort_values(by=['calc_date', 'site_id']).reset_index(drop=True)

# Drop redundant collocated devices to prevent spatial CV leakage
initial_rows = len(df)
df = df.drop_duplicates(subset=['site_id', 'calc_date'], keep='first').reset_index(drop=True)
print(f"Dropped {initial_rows - len(df)} redundant collocated device rows.")

# Dynamically recompute Kloog half-normal weights based on hours_count
deviations = (24.0 - df['hours_count']).values
formal_weights = halfnorm.pdf(deviations, loc=0, scale=10)
max_pdf_val = halfnorm.pdf(0, loc=0, scale=10)
weights_array = (formal_weights / max_pdf_val).astype('float32')

# CRITICAL LEAKAGE FIX: Added 'weight' to metadata to prevent it from leaking into features!
metadata_cols = [
    'device_name', 'calc_date', 'daily_avg_pm25', 'hours_count', 
    'latitude', 'longitude', 'site_id', 'site_name', 'lat', 'lon', 'month',
    'pm25_stage1_pred_log', 'weight'  
]
features = [c for c in df.columns if c not in metadata_cols]

print(f"Dataset successfully cleaned. Retained {len(features)} active features.")

X = df[features].astype('float32')
y_phys = df['daily_avg_pm25'].values.astype('float32')
y_log = np.log1p(y_phys)

groups_site = df['site_id'].values
dates = df['calc_date'].values

# =====================================================================
# 2. BAYESIAN HYPERPARAMETER TUNING (OPTUNA)
# =====================================================================
print("\nInitiating Bayesian Optimization via Optuna for Random Forest...")

def objective(trial):
    # Random Forest specific hyperparameter bounds
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'max_samples': trial.suggest_float('max_samples', 0.5, 0.95), # Bootstrap sampling ratio
        'bootstrap': True,
        'n_jobs': -1, # Utilize all cores
        'random_state': 42
    }
    
    # Internal evaluation using spatial folding to select parameters
    cv = GroupKFold(n_splits=4)
    rmse_scores = []
    
    for train_idx, val_idx in cv.split(X, y_log, groups=groups_site):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y_log[train_idx], y_log[val_idx]
        w_tr = weights_array[train_idx]
        
        model = RandomForestRegressor(**param)
        
        # Fit with dynamically computed spatial weights
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        
        preds = model.predict(X_va)
        fold_rmse = np.sqrt(mean_squared_error(y_va, preds))
        rmse_scores.append(fold_rmse)
        
    return np.mean(rmse_scores)

# Optimize hyper-parameters using Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30) # Fewer trials needed for RF compared to Boosting
best_params = study.best_params
print(f"\nOptimal Parameters Found: {best_params}")

# =====================================================================
# 3. TRIPLE CROSS-VALIDATION EVALUATION (Chen et al. Standard)
# =====================================================================
print("\nExecuting Final Spatiotemporal Cross-Validation Loops...")

def run_cv(splitter, X_data, y_data, weights, groups=None, is_timeseries=False):
    # Initialize zero array for true out-of-fold metrics
    pred_log_oof = np.zeros_like(y_data)
    eval_params = {**best_params, 'n_jobs': -1, 'random_state': 42}
    
    splits = splitter.split(X_data) if is_timeseries else splitter.split(X_data, y_data, groups)
    
    for tr_idx, te_idx in splits:
        fold_model = RandomForestRegressor(**eval_params)
        fold_model.fit(
            X_data.iloc[tr_idx], 
            y_data[tr_idx], 
            sample_weight=weights[tr_idx]
        )
        pred_log_oof[te_idx] = fold_model.predict(X_data.iloc[te_idx])
        
    # Mask needed strictly for TimeSeriesSplit because initial block cannot be validated against
    mask = pred_log_oof != 0
    pred_phys = np.expm1(pred_log_oof[mask])
    y_true_phys = np.expm1(y_data[mask])
    
    r2 = r2_score(y_true_phys, pred_phys)
    mae = mean_absolute_error(y_true_phys, pred_phys)
    rmse = np.sqrt(mean_squared_error(y_true_phys, pred_phys))
    mre = np.mean(np.abs((pred_phys - y_true_phys) / y_true_phys)) * 100
    return r2, mae, rmse, mre

# 1. Sample CV (Standard 10-Fold)
val_sample = run_cv(KFold(n_splits=10, shuffle=True, random_state=42), X, y_log, weights_array)

# 2. Site CV (Strictly isolating geographical testing)
val_site = run_cv(GroupKFold(n_splits=10), X, y_log, weights_array, groups=groups_site)

# 3. Time CV (Fixed Temporal Leakage using Strict Forward Time-Series Split)
# Reorganize memory tightly aligned to time axis before validation
ts_idx = np.argsort(dates)
X_ts, y_ts, weights_ts = X.iloc[ts_idx].reset_index(drop=True), y_log[ts_idx], weights_array[ts_idx]
val_time = run_cv(TimeSeriesSplit(n_splits=10), X_ts, y_ts, weights_ts, is_timeseries=True)

results = pd.DataFrame([val_sample, val_site, val_time], 
                       index=['Sample-based CV', 'Site-based CV', 'Time-based (Forward) CV'],
                       columns=['R2', 'MAE', 'RMSE', 'MRE (%)'])

print("\n" + "="*70)
print(" STAGE 2 RANDOM FOREST FINAL VALIDATION REPORT")
print("="*70)
print(results.round(4))
print("="*70)

# =====================================================================
# 4. FINAL PRODUCTION FIT & EXPORT
# =====================================================================
print("\nFitting final Stage 2 production model on full cleaned dataset...")
final_production_model = RandomForestRegressor(
    **{**best_params, 'n_jobs': -1, 'random_state': 42}
)
final_production_model.fit(X, y_log, sample_weight=weights_array)

# Export using Joblib for scikit-learn models
joblib.dump(final_production_model, MODEL_SAVE_PATH)
print(f"Success! Final Random Forest Model securely exported to: {MODEL_SAVE_PATH}")