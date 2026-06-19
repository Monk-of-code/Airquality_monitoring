import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURATION & PATHS
# =====================================================================
DATA_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stage2_training.csv'
MODEL_STG1_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stg1_xgboost_v3.json'
MODEL_STG2_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stg2_final_model.json'
OUTPUT_PLOT_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/accuracy_results/stage1_vs_stage2_consolidated_gain.png'

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 11, 'ytick.labelsize': 12, 'figure.titlesize': 18,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
})

# =====================================================================
# 2. LOAD DATASET & MAP FEATURES
# =====================================================================
print(f"Loading feature definitions from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

metadata_cols = [
    'device_name', 'calc_date', 'daily_avg_pm25', 'hours_count', 
    'latitude', 'longitude', 'site_id', 'site_name', 'lat', 'lon', 'month',
    'weight', 'pm25_stage1_pred_log'
]
features = [c for c in df.columns if c not in metadata_cols]

stg1_features = [f for f in features if f != 'improved_Ps']
stg2_features = features 

# =====================================================================
# 3. IMPORT TRAINED MODELS FROM DISK
# =====================================================================
print("\nImporting trained master models from disk...")
model_s1 = xgb.XGBRegressor()
model_s1.load_model(MODEL_STG1_PATH)

model_s2 = xgb.XGBRegressor()
model_s2.load_model(MODEL_STG2_PATH)

# =====================================================================
# 4. EXTRACT & CONSOLIDATE INFORMATION GAIN
# =====================================================================
print("\nExtracting native Information Gain and consolidating encoded groups...")

gain_raw_s1 = model_s1.get_booster().get_score(importance_type='gain')
gain_raw_s2 = model_s2.get_booster().get_score(importance_type='gain')

def get_raw_gain(gain_dict, feat_name, feat_list):
    if feat_name in gain_dict:
        return gain_dict[feat_name]
    idx = feat_list.index(feat_name) if feat_name in feat_list else -1
    return gain_dict.get(f"f{idx}", 0.0)

# Define explicit mapping from raw columns to consolidated conceptual names
# Columns mapped to a group will be summed; unmapped columns remain individual
group_mapping = {
    't1_n': 'Day of Year (Consolidated)',
    't2_n': 'Day of Year (Consolidated)',
    't3':   'Day of Year (Consolidated)',
    'viirs_SAA_cos': 'VIIRS Solar Azimuth Angle',
    'viirs_SAA_sin': 'VIIRS Solar Azimuth Angle'
}

# Consolidate Stage 1 Gains
consolidated_s1 = {}
for f in stg1_features:
    g = get_raw_gain(gain_raw_s1, f, stg1_features)
    concept_name = group_mapping.get(f, f)
    consolidated_s1[concept_name] = consolidated_s1.get(concept_name, 0.0) + g

# Consolidate Stage 2 Gains
consolidated_s2 = {}
for f in stg2_features:
    g = get_raw_gain(gain_raw_s2, f, stg2_features)
    concept_name = group_mapping.get(f, f)
    consolidated_s2[concept_name] = consolidated_s2.get(concept_name, 0.0) + g

# Align into a unified Master DataFrame
master_rows = []
# Collect all unique conceptual names
all_concepts = set(consolidated_s1.keys()).union(set(consolidated_s2.keys()))

for concept in all_concepts:
    # improved_Ps is strictly 0 in Stage 1
    g1 = consolidated_s1.get(concept, 0.0) if concept != 'improved_Ps' else 0.0
    g2 = consolidated_s2.get(concept, 0.0)
    master_rows.append({'Feature_Group': concept, 'Gain_Stage1': g1, 'Gain_Stage2': g2})

df_gains = pd.DataFrame(master_rows)

# Isolate the Top 15 dominant conceptual drivers based on Stage 2
df_top15 = df_gains.sort_values(by='Gain_Stage2', ascending=True).tail(15).reset_index(drop=True)

# =====================================================================
# 5. RENDER GROUPED HORIZONTAL BAR CHART
# =====================================================================
print("\nRendering consolidated grouped horizontal bar chart...")

fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

y_indices = np.arange(len(df_top15))
bar_height = 0.38  

# Stage 1 Bars
ax.barh(
    y_indices - bar_height/2, 
    df_top15['Gain_Stage1'], 
    height=bar_height, 
    label='Stage 1: Satellite & Met Baseline', 
    color='steelblue', 
    alpha=0.85
)

# Stage 2 Bars
ax.barh(
    y_indices + bar_height/2, 
    df_top15['Gain_Stage2'], 
    height=bar_height, 
    label='Stage 2: Spatiotemporal Corrected', 
    color='teal', 
    alpha=0.9
)

ax.set_yticks(y_indices)
ax.set_yticklabels(df_top15['Feature_Group'], fontweight='bold')
ax.set_xlabel('Consolidated Information Gain (Summed Split Contributions)', fontweight='bold', labelpad=12)
ax.set_title('Top 15 Pm2.5 Drivers: Baseline vs. Spatiotemporal Correction', fontweight='bold', pad=20)

ax.grid(True, which='major', axis='x', color='whitesmoke', linestyle='-', linewidth=1)
ax.set_axisbelow(True)
ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='lightgray', fontsize=12)

os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
print(f"✅ SUCCESS: Consolidated feature importance plot exported to:\n{OUTPUT_PLOT_PATH}")