import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import halfnorm, gaussian_kde, linregress
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURATION & EXACT DIRECTORY PATHS
# =====================================================================
DATA_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stage2_training.csv'
OUTPUT_PLOT_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/accuracy_results/stage2_site_cv_accuracy.png'

# Enforce publication sans-serif typography
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 20,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
})

# =====================================================================
# 2. DEFINITION OF PUBLICATION PLOTTING FUNCTION
# =====================================================================
def plot_publication_validation_accuracy(y_true, y_pred, output_path):
    """Generates a high-impact density scatter plot for out-of-sample validation."""
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float32).flatten()
    
    # Stacking and calculating localized point density
    xy = np.vstack([y_true, y_pred])
    z = gaussian_kde(xy)(xy)
    
    # Sort points so the densest cores are drawn on top
    idx = z.argsort()
    y_true, y_pred, z = y_true[idx], y_pred[idx], z[idx]
    
    # Statistical Metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mre = np.mean(np.abs((y_pred - y_true) / y_true)) * 100.0
    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
    
    # Draw Figure
    fig, ax = plt.subplots(figsize=(9, 9), dpi=300)
    sc = ax.scatter(y_true, y_pred, c=z, s=25, cmap='turbo', alpha=0.85, edgecolor='none')
    
    axis_min = 0.0
    axis_max = float(np.percentile(y_true, 99.5))
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    
    ax.plot([axis_min, axis_max], [axis_min, axis_max], color='gray', linestyle='--', linewidth=1.5, label='1:1 Reference Line')
    fit_line = slope * np.array([axis_min, axis_max]) + intercept
    ax.plot([axis_min, axis_max], fit_line, color='black', linestyle='-', linewidth=2.5, label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f}')
    
    metrics_text = (
        f"N = {len(y_true):,}\n"
        f"R2 = {r2:.3f}\n"
        f"MAE = {mae:.2f} ug/m3\n"
        f"RMSE = {rmse:.2f} ug/m3\n"
        f"MRE = {mre:.1f}%"
    )
    
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95, edgecolor='lightgray')
    ax.text(0.04, 0.96, metrics_text, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props, linespacing=1.4, fontweight='bold')
    
    ax.set_xlabel('Observed Ground-Level PM2.5 (ug/m3)', fontweight='bold', labelpad=12)
    ax.set_ylabel('Predicted Out-of-Sample PM2.5 (ug/m3)', fontweight='bold', labelpad=12)
    ax.set_title('Stage 2 Out-of-Sample Site Cross-Validation', fontweight='bold', pad=20)
    
    ax.grid(True, which='major', color='whitesmoke', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Point Density (Gaussian KDE)', rotation=270, labelpad=25, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='lightgray', fontsize=13)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n SUCCESS: High-resolution density validation plot exported to:\n{output_path}")

# =====================================================================
# 3. LOAD & CLEAN DATASET
# =====================================================================
print(f"Loading Stage 2 dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df['calc_date'] = pd.to_datetime(df['calc_date'])

df = df.sort_values(by=['site_id', 'calc_date', 'hours_count'], ascending=[True, True, False])
df = df.drop_duplicates(subset=['site_id', 'calc_date'], keep='first').reset_index(drop=True)
print(f"Cleaned dataset: Retained {len(df)} primary physical observations.")

deviations = (24.0 - df['hours_count']).values
formal_weights = halfnorm.pdf(deviations, loc=0, scale=10)
max_pdf_val = halfnorm.pdf(0, loc=0, scale=10)
weights_array = (formal_weights / max_pdf_val).astype('float32')

metadata_cols = [
    'device_name', 'calc_date', 'daily_avg_pm25', 'hours_count', 
    'latitude', 'longitude', 'site_id', 'site_name', 'lat', 'lon', 'month',
    'weight', 'pm25_stage1_pred_log','geom'
]
features = [c for c in df.columns if c not in metadata_cols]

X = df[features].astype('float32')
y_phys = df['daily_avg_pm25'].values.astype('float32')  # <-- Explicitly assigned here
y_log = np.log1p(y_phys)

groups_site = df['site_id'].values

# =====================================================================
# 4. EXECUTE 10-FOLD SITE-BASED CROSS VALIDATION
# =====================================================================
print("\nExecuting 10-Fold Site-based Cross-Validation...")

best_params = {
    'subsample': 0.9, 'reg_lambda': 25.0, 'n_estimators': 2000, 
    'min_child_weight': 80, 'max_depth': 3, 'learning_rate': 0.02, 
    'gamma': 0.1, 'colsample_bytree': 0.8, 'alpha': 3.0
}

build_data = xgb.build_info()
gpu_available = build_data.get('USE_CUDA', False) or (build_data.get('GPU_support', 'OFF') == 'ON')
device_config = 'cuda' if gpu_available else 'cpu'

eval_params = {**best_params, 'tree_method': 'hist', 'device': device_config, 'random_state': 42}

X_arr = np.ascontiguousarray(X.values, dtype=np.float32)
y_log_arr = np.ascontiguousarray(y_log, dtype=np.float32)
weights_arr = np.ascontiguousarray(weights_array, dtype=np.float32)

pred_log_oof = np.zeros_like(y_log_arr)
splitter = GroupKFold(n_splits=10)

for fold, (tr_idx, te_idx) in enumerate(splitter.split(X_arr, y_log_arr, groups_site), 1):
    print(f" -> Processing Fold {fold}/10...")
    fold_model = xgb.XGBRegressor(**eval_params, n_jobs=1)
    
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
mre = np.mean(np.abs((pred_phys - y_phys) / y_phys)) * 100.0

print("\n" + "="*50)
print(" STAGE 2 SITE-BASED CV ACCURACY REPORT")
print("="*50)
print(f" R2 Score : {r2:.4f}")
print(f" MAE      : {mae:.4f} ug/m3")
print(f" RMSE     : {rmse:.4f} ug/m3")
print(f" MRE (%)  : {mre:.4f} %")
print("="*50)

# =====================================================================
# 5. GENERATE PLOT
# =====================================================================
# Safely invokes the pre-defined function using variables currently in memory
plot_publication_validation_accuracy(y_phys, pred_phys, OUTPUT_PLOT_PATH)