import os
import gc
import warnings
import numpy as np
import pandas as pd
import cupy as cp
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, linregress
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURATION & PATHS
# =====================================================================
DATA_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stage2_training.csv'
OUTPUT_PLOT_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/accuracy_results/stage2_seasonal_sample_cv.png'

HARDWARE_CONFIG = {
    'device': 'cuda',
    'tree_method': 'hist'
}

# Publication-grade typography - Darkened and bolded for maximum legibility
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.titlesize': 18,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'text.color': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black',
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold'
})

# =====================================================================
# 2. DEFINITION OF SEASONAL PLOTTING FUNCTION
# =====================================================================
def plot_seasonal_accuracy(df, obs_col, pred_col, season_col, output_path):
    """Generates a 2x2 seasonal Sample-CV accuracy plot."""
    print("\nGenerating final seasonal density plots...")

    seasonal_order = ['Hot and Dry (Jan-Feb)', 'Long Rains (MAM)', 'Cool & Dry (JJA/S)', 'Short Rains (OND)']
    
    # Safely calculate a global axis limit so all 4 panels share the exact same scale
    axis_limit = float(np.percentile(df[obs_col], 99.5))
    axis_limit = max(axis_limit, float(np.percentile(df[pred_col], 99.5)))

    # 12x12 is ideal for a 2x2 grid inserted into a standard publication page
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=300, sharex=True, sharey=True)
    axes_flat = axes.ravel()
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=1.5)

    sc = None 
    for i, season in enumerate(seasonal_order):
        ax = axes_flat[i]
        sub = df[df[season_col] == season]
        
        if len(sub) < 5: 
            print(f"Skipping {season} due to insufficient data.")
            continue

        obs, pred = sub[obs_col].values, sub[pred_col].values
        
        # KDE Density (Sorting ensures density is visually layered correctly)
        xy = np.vstack([obs, pred])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        obs_s, pred_s, z_s = obs[idx], pred[idx], z[idx]
        
        # Regression metrics
        r2 = r2_score(obs, pred)
        mae = mean_absolute_error(obs, pred)
        rmse = np.sqrt(mean_squared_error(obs, pred))
        slope, intercept, _, _, _ = linregress(obs, pred)

        # Format the intercept cleanly
        intercept_str = f"+ {intercept:.2f}" if intercept >= 0 else f"- {abs(intercept):.2f}"

        # 1. Scatter Plot
        sc = ax.scatter(obs_s, pred_s, c=z_s, s=25, cmap='turbo', alpha=0.85, edgecolor='none')
        
        # 2. Reference Lines
        ax.plot([0, axis_limit], [0, axis_limit], color='dimgray', linestyle='--', lw=1.5, label='1:1 Line')
        ax.plot([0, axis_limit], intercept + slope * np.array([0, axis_limit]), color='black', lw=2.5, label=f'Fit: y = {slope:.2f}x {intercept_str}')
        
        # 3. Top-Left Box: Statistical Metrics Only
        stats_text = (f"N = {len(obs):,}\n"
                      f"R² = {r2:.3f}\n"
                      f"MAE = {mae:.2f} µg/m³\n"
                      f"RMSE = {rmse:.2f} µg/m³")
        
        ax.text(0.04, 0.96, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='left', 
                bbox=props, fontweight='bold', color='black')
        
        # 4. Bottom-Right Box: Equation and Legend
        legend = ax.legend(loc='lower right', framealpha=0.95, edgecolor='gray', fancybox=True, prop={'weight': 'bold', 'size': 11})
        legend.get_frame().set_linewidth(1.5)
        for text in legend.get_texts(): text.set_color("black")
                
        ax.set_title(season, fontweight='bold', pad=12)
        ax.set_xlim(0, axis_limit); ax.set_ylim(0, axis_limit)

    fig.supxlabel('Observed Ground-Level PM2.5 (µg/m³)', fontweight='bold', fontsize=16)
    fig.supylabel('Predicted Out-of-Fold PM2.5 (µg/m³)', fontweight='bold', fontsize=16)
    fig.suptitle('Stage 2 Accuracy by Season', fontweight='bold', y=0.96)

    if sc:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
        cbar.set_label('Point Density (Gaussian KDE)', fontweight='bold', color='black', rotation=270, labelpad=20)
        cbar.ax.yaxis.set_tick_params(color='black', labelcolor='black')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\n SUCCESS: Publication-grade seasonal plot saved to:\n{output_path}")


# =====================================================================
# 3. DATA PREPARATION & SEASON MAPPING
# =====================================================================
print(f"Loading Stage 2 dataset from: {DATA_PATH}")
df_stage2 = pd.read_csv(DATA_PATH)
df_stage2['calc_date'] = pd.to_datetime(df_stage2['calc_date'])
df_stage2['month'] = df_stage2['calc_date'].dt.month

# Clean duplicates & apply 16-hour cutoff
df_stage2 = df_stage2.sort_values(by=['site_id', 'calc_date', 'hours_count'], ascending=[True, True, False])
df_stage2 = df_stage2.drop_duplicates(subset=['site_id', 'calc_date'], keep='first').reset_index(drop=True)
df_stage2 = df_stage2[df_stage2['hours_count'] >= 16].reset_index(drop=True)

# Synchronized Seasonal Mapping
SEASON_MAP = {
    1: 'Hot and Dry (Jan-Feb)', 2: 'Hot and Dry (Jan-Feb)',
    3: 'Long Rains (MAM)', 4: 'Long Rains (MAM)', 5: 'Long Rains (MAM)',
    6: 'Cool & Dry (JJA/S)', 7: 'Cool & Dry (JJA/S)', 8: 'Cool & Dry (JJA/S)', 9: 'Cool & Dry (JJA/S)',
    10: 'Short Rains (OND)', 11: 'Short Rains (OND)', 12: 'Short Rains (OND)'
}
df_stage2['season'] = df_stage2['month'].map(SEASON_MAP)

y_target = df_stage2['daily_avg_pm25'].values.astype(np.float32)
weights_array = df_stage2['weight'].values.astype(np.float32) if 'weight' in df_stage2.columns else np.ones(len(y_target), dtype=np.float32)

metadata_cols = [
    'device_name', 'calc_date', 'daily_avg_pm25', 'hours_count', 'latitude', 'longitude', 
    'site_id', 'site_name', 'lat', 'lon', 'weight', 'month', 'geom', 'sub_county',
    'pm25_stage1_pred_log', 'pm25_stage1_pred_log_OOF', 'pm25_stage1_pred_physical_OOF', 
    'pm25_stage1_pred_physical', 'season'
]

features = [c for c in df_stage2.columns if c not in metadata_cols]

# Stage 2 uses all features (including improved_Ps)
X_stg2 = np.ascontiguousarray(df_stage2[features].values, dtype=np.float32)
y_arr = np.ascontiguousarray(y_target, dtype=np.float32)
w_arr = np.ascontiguousarray(weights_array, dtype=np.float32)

# =====================================================================
# 4. HYPERPARAMETERS
# =====================================================================
best_params_stg2 = {
    'subsample': 0.90, 'reg_lambda': 25.0, 'n_estimators': 2000,
    'min_child_weight': 80, 'max_depth': 3, 'learning_rate': 0.02,
    'gamma': 0.1, 'colsample_bytree': 0.8, 'alpha': 3.0
}

eval_params_stg2 = {**best_params_stg2, **HARDWARE_CONFIG, 'objective': 'reg:squarederror', 'random_state': 42}


# =====================================================================
# 5. FAST 10-FOLD SAMPLE-BASED CV LOOP (GPU ACCELERATED)
# =====================================================================
print("\nExecuting Fast 10-Fold Sample-Based CV for Stage 2 OOF Predictions...")

pred_oof = np.zeros(len(df_stage2), dtype=np.float32)
splitter = KFold(n_splits=10, shuffle=True, random_state=42)

for fold, (tr_idx, te_idx) in enumerate(splitter.split(X_stg2, y_arr), 1):
    print(f"  -> Training Fold {fold}/10...")
    
    # 1. Move slice to GPU VRAM
    X_tr_gpu = cp.array(X_stg2[tr_idx])
    y_tr_gpu = cp.array(y_arr[tr_idx])
    w_tr_gpu = cp.array(w_arr[tr_idx])
    X_te_gpu = cp.array(X_stg2[te_idx])
    
    # 2. Fit model
    fold_model = xgb.XGBRegressor(**eval_params_stg2)
    fold_model.fit(X_tr_gpu, y_tr_gpu, sample_weight=w_tr_gpu)
    
    # 3. Predict and pull back to CPU
    preds_gpu = fold_model.predict(X_te_gpu)
    pred_oof[te_idx] = cp.asnumpy(preds_gpu)
    
    # 4. Aggressive VRAM Cleanup
    del fold_model, X_tr_gpu, y_tr_gpu, w_tr_gpu, X_te_gpu, preds_gpu
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

# Prevent negative artifacts
df_stage2['oof_pred_phys'] = np.clip(pred_oof, a_min=0, a_max=None)


# =====================================================================
# 6. GENERATE FINAL SEASONAL PLOT
# =====================================================================
plot_seasonal_accuracy(
    df=df_stage2, 
    obs_col='daily_avg_pm25', 
    pred_col='oof_pred_phys', 
    season_col='season', 
    output_path=OUTPUT_PLOT_PATH
)