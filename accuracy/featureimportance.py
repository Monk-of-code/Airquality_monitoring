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
MODEL_STG1_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stg1_xgboost_v3.json'
MODEL_STG2_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_stage2_production_model.json'
OUTPUT_PLOT_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/accuracy_results/stage1_vs_stage2_feature_importance.png'

# Enforce clean, dark sans-serif typography suitable for publication
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'figure.titlesize': 18,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'text.color': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black',
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold'
})

# Dictionary to unify encoded/lagged variables (Angles removed since they are dropped)
GROUP_MAP = {
    # Temporal & Meteorological Groupings
    'Total_Precipitation_mm': 'Precipitation',
    'Precip_3d_sum': 'Precipitation',
    'PBL_Height': 'PBL Height',
    'PBL_Height_lag1': 'PBL Height',
    'Relative_Humidity': 'Relative Humidity',
    'Relative_Humidity_lag1': 'Relative Humidity',
    
    # Wind Components Grouping
    'Wind_Speed_10m': 'Wind Speed (10m)',
    
    # Day of Year encodings
    't1_n': 'Day of Year',
    't2_n': 'Day of Year',
    't3': 'Day of Year',
    
    # Clean up names for presentation
    'improved_Ps': 'Spatial Lag (Ps)',
    'pop_density': 'Population Density',
    'road_density': 'Road Density',
    'Temp_2m': 'Temperature (2m)',
    'Surface_Pressure': 'Surface Pressure',
    'building_heights': 'Building Heights',
    'industry_influence': 'Industry Influence',
    'transit_influence': 'Transit Influence',
    'waste_influence': 'Waste Influence'
}

# Explicitly dropping ALL angles and unwanted interaction terms
UNWANTED_FEATURES = [
    'st_interact_1', 'st_interact_2',
    'viirs_VAA', 'viirs_VAA_cos', 'viirs_VAA_sin',
    'viirs_VZA', 'viirs_VZA_cos', 'viirs_VZA_sin',
    'viirs_SAA', 'viirs_SAA_cos', 'viirs_SAA_sin',
    'viirs_SZA', 'viirs_SZA_cos', 'viirs_SZA_sin'
]

# =====================================================================
# 2. FEATURE EXTRACTION & PROCESSING FUNCTION
# =====================================================================
def get_top_features(model_path, top_n=20):
    """Loads a model, groups features, normalizes to 10, and returns top N."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"🚨 Model not found at {model_path}")

    # Load model
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    # Extract Gain importance
    imp_dict = model.get_booster().get_score(importance_type='gain')
    df = pd.DataFrame(list(imp_dict.items()), columns=['Feature', 'Raw_Gain'])

    # Filter out explicitly unwanted features
    df = df[~df['Feature'].isin(UNWANTED_FEATURES)].copy()

    # Group Encoded Features
    df['Grouped_Feature'] = df['Feature'].map(GROUP_MAP).fillna(df['Feature'])
    df_grouped = df.groupby('Grouped_Feature', as_index=False)['Raw_Gain'].sum()

    # Normalize X-axis from 0 to 10
    max_gain = df_grouped['Raw_Gain'].max()
    df_grouped['Norm_Gain'] = (df_grouped['Raw_Gain'] / max_gain) * 10.0

    # Extract Top N and sort ascending for horizontal bar plot
    df_top = df_grouped.nlargest(top_n, 'Norm_Gain').sort_values(by='Norm_Gain', ascending=True).reset_index(drop=True)
    
    return df_top

# =====================================================================
# 3. PROCESS BOTH MODELS
# =====================================================================
print("\nExtracting & Grouping Feature Importances...")
df_stg1 = get_top_features(MODEL_STG1_PATH, top_n=20)
df_stg2 = get_top_features(MODEL_STG2_PATH, top_n=20)

# =====================================================================
# 4. GENERATE SIDE-BY-SIDE PLOT
# =====================================================================
print("Generating 1x2 publication plot...")

# 18x10 figure comfortably fits 20 features
fig, axes = plt.subplots(1, 2, figsize=(18, 10), dpi=300, sharex=True)

# -----------------------------------------------------------
# PANEL A: STAGE 1
# -----------------------------------------------------------
ax1 = axes[0]
bars1 = ax1.barh(df_stg1['Grouped_Feature'], df_stg1['Norm_Gain'], 
                 color='#5C93C4', edgecolor='black', linewidth=1.2, height=0.7)

ax1.set_title('(a) Stage 1: Base Feature Importance', fontweight='bold', pad=15)
ax1.set_xlabel('Feature Importance (Gain)', fontweight='bold', labelpad=12)

# -----------------------------------------------------------
# PANEL B: STAGE 2
# -----------------------------------------------------------
ax2 = axes[1]
bars2 = ax2.barh(df_stg2['Grouped_Feature'], df_stg2['Norm_Gain'], 
                 color='#1D8A88', edgecolor='black', linewidth=1.2, height=0.7)

# Highlight Spatial Autocorrelation dynamically if it exists in the top 20
'''for idx, feature in enumerate(df_stg2['Grouped_Feature']):
    if feature == 'Spatial Autocorrelation (Ps)':
        bars2[idx].set_color('crimson')
        bars2[idx].set_edgecolor('black')'''

ax2.set_title('(b) Stage 2: Corrected Feature Importance', fontweight='bold', pad=15)
ax2.set_xlabel(' Feature Importance (Gain)', fontweight='bold', labelpad=12)

# -----------------------------------------------------------
# SHARED FORMATTING
# -----------------------------------------------------------
for ax in axes:
    ax.set_xlim(0, 10.2) # Tight padding since labels are removed
    ax.set_xticks(np.arange(0, 11, 2))
    ax.grid(axis='x', linestyle='--', alpha=0.7, color='gray')
    ax.set_axisbelow(True)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

fig.suptitle('Feature Importance Evolution: Top 20 Drivers by Gain', fontsize=20, fontweight='bold', y=1.02)

# Adjust layout to prevent y-axis label clipping
plt.tight_layout()

# Save
os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PLOT_PATH)), exist_ok=True)
plt.savefig(OUTPUT_PLOT_PATH, bbox_inches='tight')
print(f"✅ SUCCESS: Dual feature importance plot exported to:\n{OUTPUT_PLOT_PATH}")