import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD YOUR CURRENT MODEL
print("Loading the optimized XGBoost model...")
model_path = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_xgboost.json'

best_model = xgb.XGBRegressor()
best_model.load_model(model_path)

# 2. MATCH THE EXACT COLUMNS FROM YOUR SCRIPT
final_columns = [
    'month', 'day_of_week',
    'industry_influence', 'transit_influence', 'pop_density', 
    'road_density', 'building_heights', 'elevation',
    'ndvi', 'viirs_M03', 'viirs_M11', 'viirs_SZA', 
    'viirs_SAA', 'viirs_VZA', 'viirs_VAA',
    'Total_Precipitation_mm', 'U_Wind_10m', 'V_Wind_10m', 
    'Relative_Humidity', 'PBL_Height'
]

# 3. EXTRACT IMPORTANCE (GAIN)
booster = best_model.get_booster()
importance_dict = booster.get_score(importance_type='gain')

# Map the scores to the actual feature names
importance_df = pd.DataFrame({
    'Feature': list(importance_dict.keys()),
    'Importance_Gain': list(importance_dict.values())
})

# XGBoost uses internal names like 'f0', 'f1'. Let's map them back to your column names
feature_map = {f"f{i}": col for i, col in enumerate(final_columns)}
importance_df['Feature'] = importance_df['Feature'].map(feature_map).fillna(importance_df['Feature'])

# Sort from most important to least important
importance_df = importance_df.sort_values(by='Importance_Gain', ascending=False).reset_index(drop=True)

# Print the tabular results to the terminal
print("\n🏆 FEATURE IMPORTANCE RANKING (By Gain):")
print("-" * 50)
for index, row in importance_df.iterrows():
    print(f"{index + 1}. {row['Feature']:<25} : {row['Importance_Gain']:.2f}")

# 4. GENERATE PUBLICATION-READY PLOT
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

ax = sns.barplot(x='Importance_Gain', y='Feature', data=importance_df, palette='viridis')

plt.title('XGBoost Feature Importance (Gain) for Nairobi PM$_{2.5}$', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Relative Information Gain', fontsize=14)
plt.ylabel('Predictor Variables', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

for p in ax.patches:
    width = p.get_width()
    plt.text(width + (importance_df['Importance_Gain'].max() * 0.01), 
             p.get_y() + p.get_height() / 2, 
             f'{width:.1f}', 
             ha='left', va='center', fontsize=10)

plt.tight_layout()
output_image = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_feature_importance.png'
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"\n✅ High-resolution plot saved to: {output_image}")