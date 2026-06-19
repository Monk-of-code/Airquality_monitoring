import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import halfnorm, gaussian_kde, linregress
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURATION & PATHS
# =====================================================================
DATA_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stage2_training.csv'
OUTPUT_PLOT_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/accuracy_results/stage1_vs_stage2_honest_cv.png'

# Enforce clean sans-serif typography suitable for panels and publication
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.titlesize': 18,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
})

# =====================================================================
# 2. DEFINITION OF DUAL PLOTTING FUNCTION
# =====================================================================
def plot_dual_validation_accuracy(y_true, y_pred_stg1, y_pred_stg2, output_path):
    """Generates an honest 1x2 side-by-side out-of-fold CV comparison plot."""
    print("\nComputing Gaussian Kernel Density Estimates (KDE) for both stages...")
    
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_pred_stg1 = np.asarray(y_pred_stg1, dtype=np.float32).flatten()
    y_pred_stg2 = np.asarray(y_pred_stg2, dtype=np.float32).flatten()
    
    # KDE Stage 1
    xy1 = np.vstack([y_true, y_pred_stg1])
    z1 = gaussian_kde(xy1)(xy1)
    idx1 = z1.argsort()
    y_true1, y_pred1, z1 = y_true[idx1], y_pred_stg1[idx1], z1[idx1]
    
    # KDE Stage 2
    xy2 = np.vstack([y_true, y_pred_stg2])
    z2 = gaussian_kde(xy2)(xy2)
    idx2 = z2.argsort()
    y_true2, y_pred2, z2 = y_true[idx2], y_pred_stg2[idx2], z2[idx2]
    
    # Metrics Stage 1
    r2_1 = r2_score(y_true, y_pred_stg1)
    mae_1 = mean_absolute_error(y_true, y_pred_stg1)
    rmse_1 = np.sqrt(mean_squared_error(y_true, y_pred_stg1))
    mre_1 = np.mean(np.abs((y_pred_stg1 - y_true) / y_true)) * 100.0
    slope1, intercept1, _, _, _ = linregress(y_true, y_pred_stg1)
    
    # Metrics Stage 2
    r2_2 = r2_score(y_true, y_pred_stg2)
    mae_2 = mean_absolute_error(y_true, y_pred_stg2)
    rmse_2 = np.sqrt(mean_squared_error(y_true, y_pred_stg2))
    mre_2 = np.mean(np.abs((y_pred_stg2 - y_true) / y_true)) * 100.0
    slope2, intercept2, _, _, _ = linregress(y_true, y_pred_stg2)
    
    axis_min = 0.0
    axis_max = float(np.percentile(y_true, 99.5))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300, sharey=True)
    
    # --- PANEL A: STAGE 1 (HONEST OUT-OF-FOLD) ---
    ax1 = axes[0]
    sc1 = ax1.scatter(y_true1, y_pred1, c=z1, s=20, cmap='turbo', alpha=0.85, edgecolor='none')
    ax1.set_xlim(axis_min, axis_max)
    ax1.set_ylim(axis_min, axis_max)
    ax1.plot([axis_min, axis_max], [axis_min, axis_max], color='gray', linestyle='--', linewidth=1.5)
    fit_line1 = slope1 * np.array([axis_min, axis_max]) + intercept1
    ax1.plot([axis_min, axis_max], fit_line1, color='black', linestyle='-', linewidth=2, label=f'Fit: y = {slope1:.2f}x + {intercept1:.2f}')
    
    metrics_stg1 = (
        f"Stage 1 \n"
        f"N = {len(y_true):,}\n"
        f"R² = {r2_1:.4f}\n"
        f"MAE = {mae_1:.2f} µg/m³\n"
        f"RMSE = {rmse_1:.2f} µg/m³\n"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='lightgray')
    ax1.text(0.04, 0.96, metrics_stg1, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=props, linespacing=1.4, fontweight='bold')
    ax1.set_xlabel('Observed Ground-Level PM2.5 (µg/m³)', fontweight='bold', labelpad=10)
    ax1.set_ylabel('Predicted Out-of-Fold PM2.5 (µg/m³)', fontweight='bold', labelpad=10)
    ax1.set_title('(a) Stage 1: Satellite and Met Only ', fontweight='bold', pad=15)
    ax1.grid(True, which='major', color='whitesmoke', linestyle='-', linewidth=1)
    ax1.set_axisbelow(True)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='lightgray')
    
    # --- PANEL B: STAGE 2 (HONEST OUT-OF-FOLD) ---
    ax2 = axes[1]
    sc2 = ax2.scatter(y_true2, y_pred2, c=z2, s=20, cmap='turbo', alpha=0.85, edgecolor='none')
    ax2.set_xlim(axis_min, axis_max)
    ax2.set_ylim(axis_min, axis_max)
    ax2.plot([axis_min, axis_max], [axis_min, axis_max], color='gray', linestyle='--', linewidth=1.5)
    fit_line2 = slope2 * np.array([axis_min, axis_max]) + intercept2
    ax2.plot([axis_min, axis_max], fit_line2, color='black', linestyle='-', linewidth=2, label=f'Fit: y = {slope2:.2f}x + {intercept2:.2f}')
    
    metrics_stg2 = (
        f"Stage 2 \n"
        f"N = {len(y_true):,}\n"
        f"R² = {r2_2:.4f}\n"
        f"MAE = {mae_2:.2f} µg/m³\n"
        f"RMSE = {rmse_2:.2f} µg/m³\n"
        
    )
    ax2.text(0.04, 0.96, metrics_stg2, transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=props, linespacing=1.4, fontweight='bold')
    ax2.set_xlabel('Observed Ground-Level PM2.5 (µg/m³)', fontweight='bold', labelpad=10)
    ax2.set_title('(b) Stage 2: Spatiotemporal Lag (Ps) Added', fontweight='bold', pad=15)
    ax2.grid(True, which='major', color='whitesmoke', linestyle='-', linewidth=1)
    ax2.set_axisbelow(True)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='lightgray')
    
    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist(), fraction=0.02, pad=0.03)
    cbar.set_label('Point Density (Gaussian KDE)', rotation=270, labelpad=20, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    fig.suptitle('Model comparison: Baseline vs. Spatiotemporal Correction', fontsize=18, fontweight='bold', y=1.02)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n SUCCESS: Honest side-by-side CV plot exported to:\n{output_path}")

# =====================================================================
# 3. LOAD & CLEAN DATASET
# =====================================================================
print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df['calc_date'] = pd.to_datetime(df['calc_date'])

df = df.sort_values(by=['site_id', 'calc_date', 'hours_count'], ascending=[True, True, False])
df = df.drop_duplicates(subset=['site_id', 'calc_date'], keep='first').reset_index(drop=True)

deviations = (24.0 - df['hours_count']).values
formal_weights = halfnorm.pdf(deviations, loc=0, scale=10)
weights_array = (formal_weights / halfnorm.pdf(0, loc=0, scale=10)).astype('float32')

metadata_cols = [
    'device_name', 'calc_date', 'daily_avg_pm25', 'hours_count', 'latitude', 'longitude', 
    'site_id', 'site_name', 'lat', 'lon', 'weight', 'month', 'geom', 'sub_county',
    'pm25_stage1_pred_log', 'pm25_stage1_pred_log_OOF', 'pm25_stage1_pred_physical_OOF', 
    'pm25_stage1_pred_physical', 'season'
]
features = [c for c in df.columns if c not in metadata_cols]

y_phys = df['daily_avg_pm25'].values.astype('float32')
y_log = np.log1p(y_phys)
# Groups site is no longer needed for KFold, but kept to avoid variable reference errors if used elsewhere
groups_site = df['site_id'].values

# Isolate feature sets explicitly
stg1_features = [f for f in features if f != 'improved_Ps']
stg2_features = features  # Includes improved_Ps

X_stg1 = np.ascontiguousarray(df[stg1_features].values, dtype=np.float32)
X_stg2 = np.ascontiguousarray(df[stg2_features].values, dtype=np.float32)
y_log_arr = np.ascontiguousarray(y_log, dtype=np.float32)
weights_arr = np.ascontiguousarray(weights_array, dtype=np.float32)

# =====================================================================
# 4. DEFINE SEPARATE OPTIMAL HYPERPARAMETERS FOR BOTH STAGES
# =====================================================================
build_data = xgb.build_info()
gpu_avail = build_data.get('USE_CUDA', False) or (build_data.get('GPU_support', 'OFF') == 'ON')
device_setting = 'cuda' if gpu_avail else 'cpu'

# ---------------------------------------------------------------------
#  REPLACE THESE WITH YOUR EXACT STAGE 1 TUNED PARAMETERS
# ---------------------------------------------------------------------
best_params_stg1 = {
    'n_estimators': 1500,
    'learning_rate': 0.005,
    'max_depth': 8,
    'min_child_weight': 30,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'gamma': 0.5,
    'alpha': 0.1,
    'reg_lambda': 10.0
}

# ---------------------------------------------------------------------
# STAGE 2 TUNED PARAMETERS (Confirmed crossing 0.70 threshold)
# ---------------------------------------------------------------------
best_params_stg2 = {
    'subsample': 0.8,
    'reg_lambda': 20.0, 
    'n_estimators':1000,
    'min_child_weight': 50,
    'max_depth': 5,
    'learning_rate': 0.01,
    'gamma': 0.5, 
    'colsample_bytree': 0.8, 
    'alpha': 0.1
}

eval_params_stg1 = {**best_params_stg1, 'tree_method': 'hist', 'device': device_setting, 'random_state': 42}
eval_params_stg2 = {**best_params_stg2, 'tree_method': 'hist', 'device': device_setting, 'random_state': 42}

# =====================================================================
# 5. EXECUTE HONEST DUAL 10-FOLD CV LOOPS (Sample Based)
# =====================================================================
print("\nExecuting rigorous 10-Fold Sample-based CV using separate tuned parameters...")

pred_log_oof_stg1 = np.zeros_like(y_log_arr)
pred_log_oof_stg2 = np.zeros_like(y_log_arr)

# Changed from GroupKFold to KFold
splitter = KFold(n_splits=10, shuffle=True, random_state=42)

for fold, (tr_idx, te_idx) in enumerate(splitter.split(X_stg1, y_log_arr), 1):
    print(f" -> Processing Fold {fold}/10 (Evaluating both stages out-of-fold)...")
    
    # --- Train Stage 1 Out-of-Fold with Stage 1 Best Params ---
    model_s1 = xgb.XGBRegressor(**eval_params_stg1, n_jobs=1)
    model_s1.fit(np.ascontiguousarray(X_stg1[tr_idx]), np.ascontiguousarray(y_log_arr[tr_idx]), 
                 sample_weight=np.ascontiguousarray(weights_arr[tr_idx]))
    pred_log_oof_stg1[te_idx] = model_s1.predict(np.ascontiguousarray(X_stg1[te_idx]))
    
    # --- Train Stage 2 Out-of-Fold with Stage 2 Best Params ---
    model_s2 = xgb.XGBRegressor(**eval_params_stg2, n_jobs=1)
    model_s2.fit(np.ascontiguousarray(X_stg2[tr_idx]), np.ascontiguousarray(y_log_arr[tr_idx]), 
                 sample_weight=np.ascontiguousarray(weights_arr[tr_idx]))
    pred_log_oof_stg2[te_idx] = model_s2.predict(np.ascontiguousarray(X_stg2[te_idx]))

pred_phys_stg1 = np.expm1(pred_log_oof_stg1)
pred_phys_stg2 = np.expm1(pred_log_oof_stg2)

# =====================================================================
# 6. GENERATE HONEST COMPARISON PLOT
# =====================================================================
plot_dual_validation_accuracy(y_phys, pred_phys_stg1, pred_phys_stg2, OUTPUT_PLOT_PATH)