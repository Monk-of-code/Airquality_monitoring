# =============================================================================
# Nairobi PM2.5 – GAM + RF Stacked Model
# Update: Natively uses Latitude/Longitude, explicit GAM alignment
# =============================================================================
import pandas as pd
import numpy as np
import warnings
import time
from pygam import LinearGAM, te, l
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. LOAD DATA & WEIGHTS
# ---------------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv("/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_ml_cleaned_training_dataset.csv")
df["calc_date"] = pd.to_datetime(df["calc_date"])

# Ensure no completely missing coordinates
df = df.dropna(subset=['latitude', 'longitude'])

# Half-normal weights
df["hours_count"] = df["hours_count"].clip(upper=24)
sigma = 10
weights = np.exp(-((24 - df["hours_count"]) ** 2) / (2 * sigma**2)).astype("float32")

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
print("Engineering features...")

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

# Combine U and V wind into overall Wind Speed
df["Wind_Speed"] = np.sqrt(df["U_Wind_10m"] ** 2 + df["V_Wind_10m"] ** 2)

# Cloud/Missing Proxies
df["NO2_avail"] = df["NO2"].notna().astype(int)
df["SO2_avail"] = df["SO2"].notna().astype(int)
df["CO_avail"] = df["CO"].notna().astype(int)
df["M03_avail"] = df["viirs_M03"].notna().astype(int)
df["M11_avail"] = df["viirs_M11"].notna().astype(int)

# ---------------------------------------------------------------------------
# 3. FEATURE SETS (EXACTLY 8 GAM FEATURES)
# ---------------------------------------------------------------------------
gam_features = [
    "longitude", "latitude",      # Indices: 0, 1
    "month_sin", "month_cos",     # Indices: 2, 3
    "Wind_Speed",                 # Index: 4
    "Relative_Humidity",          # Index: 5
    "PBL_Height",                 # Index: 6
    "Temp_2m"                     # Index: 7
]

rf_features = [
    "longitude", "latitude", "month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos",
    "industry_influence", "transit_influence", "pop_density", "road_density", 
    "building_heights", "elevation", "ndvi", "viirs_M03", "viirs_M11", "viirs_SZA", 
    "viirs_VZA", "viirs_SAA_sin", "viirs_SAA_cos", "viirs_VAA_sin", "viirs_VAA_cos",
    "NO2", "CO", "SO2", "Total_Precipitation_mm", "U_Wind_10m", "V_Wind_10m",
    "Wind_Speed", "Relative_Humidity", "PBL_Height", "Temp_2m",
    "NO2_avail", "SO2_avail", "CO_avail", "M03_avail", "M11_avail"
]

X_gam = df[gam_features]
X_rf = df[rf_features]
y_all = np.log1p(df["daily_avg_pm25"])
groups = df["site_id"]

# ---------------------------------------------------------------------------
# PHASE 1: GLOBAL SPATIAL TUNING (Run Once)
# ---------------------------------------------------------------------------
print("\n=== PHASE 1: Global Hyperparameter Tuning ===")
t0 = time.time()

# Fit GAM on entire dataset to get global residuals for tuning
gam_imputer_global = SimpleImputer(strategy='median')
X_gam_clean_global = gam_imputer_global.fit_transform(X_gam)

# GAM properly aligned to the 8 columns (0 through 7)
gam_global = LinearGAM(
    te(0, 1, n_splines=[10, 10]) + # Tensor product of longitude (0) and latitude (1)
    te(2, 3, n_splines=[8, 8]) +   # Tensor product of month_sin (2) and month_cos (3)
    l(4) + l(5) + l(6) + l(7)      # Linear terms for the remaining 4 weather variables
)
gam_global.fit(X_gam_clean_global, y_all, weights=weights) 
global_residuals = y_all - gam_global.predict(X_gam_clean_global)

# Impute RF features for tuning
rf_imputer_global = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
X_rf_imp_global = rf_imputer_global.fit_transform(X_rf).astype("float32")

# Tune RF on the global residuals
param_grid = {
    "n_estimators": [500, 800, 1000],
    "max_depth": [15, 20, 25, None],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 5, 10],
    "max_features": ["sqrt", 0.33],
    "max_samples": [0.7, 0.8, 0.9]
}

search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_grid,
    n_iter=20,
    scoring="r2",
    cv=GroupKFold(n_splits=5),
    random_state=42,
    n_jobs=1
)
search.fit(X_rf_imp_global, global_residuals, groups=groups, sample_weight=weights)
best_rf_params = search.best_params_

print(f"Tuning complete in {(time.time() - t0)/60:.1f} minutes.")
print("Locked Parameters:", best_rf_params)

# ---------------------------------------------------------------------------
# PHASE 2: EVALUATION HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def fit_and_predict_stack(train_idx, test_idx):
    """Handles strict data isolation, imputation, and prediction for a single fold."""
    X_gam_tr, X_gam_te = X_gam.iloc[train_idx], X_gam.iloc[test_idx]
    X_rf_tr, X_rf_te = X_rf.iloc[train_idx], X_rf.iloc[test_idx]
    y_tr = y_all.iloc[train_idx]
    w_tr = weights.iloc[train_idx]
    
    # GAM Isolation
    gam_imputer = SimpleImputer(strategy='median')
    X_g_tr_imp = gam_imputer.fit_transform(X_gam_tr)
    X_g_te_imp = gam_imputer.transform(X_gam_te)
    
    # GAM properly aligned
    gam = LinearGAM(
        te(0, 1, n_splines=[10, 10]) + te(2, 3, n_splines=[8, 8]) + 
        l(4) + l(5) + l(6) + l(7)
    )
    gam.fit(X_g_tr_imp, y_tr, weights=w_tr)
    residuals_tr = y_tr - gam.predict(X_g_tr_imp)
    
    # RF Isolation
    rf_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
    X_r_tr_imp = rf_imputer.fit_transform(X_rf_tr).astype("float32")
    X_r_te_imp = rf_imputer.transform(X_rf_te).astype("float32")
    
    rf = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
    rf.fit(X_r_tr_imp, residuals_tr, sample_weight=w_tr)
    
    # Predict
    trend_te = gam.predict(X_g_te_imp)
    resid_te = rf.predict(X_r_te_imp)
    
    return trend_te + resid_te

# ---------------------------------------------------------------------------
# EVALUATION 1: 80/20 Holdout
# ---------------------------------------------------------------------------
print("\n=== EVALUATION 1: 80/20 Holdout ===")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_gam, y_all, groups))

pred_log = fit_and_predict_stack(train_idx, test_idx)
y_true = np.expm1(y_all.iloc[test_idx])
y_pred = np.expm1(pred_log)

print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f} µg/m³")
print(f"R²  : {r2_score(y_true, y_pred):.3f}")

# ---------------------------------------------------------------------------
# EVALUATION 2: 10-Fold CV
# ---------------------------------------------------------------------------
print("\n=== EVALUATION 2: 10-Fold Spatial CV ===")
gkf10 = GroupKFold(n_splits=10)
pred_log_cv = np.zeros(len(y_all), dtype="float32")

for fold, (tr_idx, te_idx) in enumerate(gkf10.split(X_gam, y_all, groups), 1):
    pred_log_cv[te_idx] = fit_and_predict_stack(tr_idx, te_idx)

y_true_cv = np.expm1(y_all)
y_pred_cv = np.expm1(pred_log_cv)
print(f"10-Fold RMSE: {np.sqrt(mean_squared_error(y_true_cv, y_pred_cv)):.2f} µg/m³")
print(f"10-Fold R²  : {r2_score(y_true_cv, y_pred_cv):.3f}")

# ---------------------------------------------------------------------------
# EVALUATION 3: Sparse-Site (<30 days)
# ---------------------------------------------------------------------------
print("\n=== EVALUATION 3: Sparse-Site (<30 days) ===")
site_counts = df["site_id"].value_counts()
sparse_sites = site_counts[site_counts < 30].index.tolist()

sparse_res = []
for site in sparse_sites:
    train_mask = np.where(df["site_id"] != site)[0]
    test_mask = np.where(df["site_id"] == site)[0]
    
    if len(test_mask) < 2: continue
    
    pred_log = fit_and_predict_stack(train_mask, test_mask)
    y_true_s = np.expm1(y_all.iloc[test_mask])
    y_pred_s = np.expm1(pred_log)
    
    rmse_s = np.sqrt(mean_squared_error(y_true_s, y_pred_s))
    sparse_res.append({"site_id": site, "days": len(test_mask), "RMSE": rmse_s})
    print(f"{site[:15]:<15} ({len(test_mask):>2} days) | RMSE: {rmse_s:.2f} µg/m³")

sparse_df = pd.DataFrame(sparse_res)
print(f"\nMean Sparse RMSE: {sparse_df['RMSE'].mean():.2f} µg/m³")