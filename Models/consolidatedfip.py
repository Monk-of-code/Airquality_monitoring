import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# 1. Load the model from your local path
model = xgb.XGBRegressor()
model.load_model('/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_xgboost_v2_no_month.json')

# 2. Get the raw Gain scores
importance = model.get_booster().get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'Feature': list(importance.keys()),
    'Gain': list(importance.values())
})

# 3. Define the custom grouping logic
def group_presentation_features(name):
    # Group Monthly cyclic features
    if 'month' in name:
        return 'Month'
    # Group Day of week cyclic features
    if 'day_of_week' in name or 'day_of_the_week' in name:
        return 'Day of the Week'
    # Group Vector Azimuth Angles (VAA)
    if 'viirs_VAA' in name:
        return 'viirs_VAA'
    # Group Solar Azimuth Angles (SAA)
    if 'viirs_SAA' in name:
        return 'viirs_SAA'
    
    # Optional: Clean up other names for professional display
    clean_names = {
        'building_heights': 'Building Heights',
        'Temp_2m': 'Temperature',
        'PBL_Height': 'Boundary Layer Height',
        'Relative_Humidity': 'Relative Humidity',
        'NO2': 'Satellite NO2',
        'industry_influence': 'Industrial Proximity',
        'viirs_SZA': 'Solar Zenith Angle (SZA)',
        'viirs_VZA': 'Viewing Zenith Angle (VZA)'
    }
    return clean_names.get(name, name)

importance_df['Display_Name'] = importance_df['Feature'].apply(group_presentation_features)

# 4. Aggregate by the new Display_Name
grouped_df = importance_df.groupby('Display_Name')['Gain'].sum().sort_values(ascending=False).reset_index()

# 5. Create the Plot
plt.figure(figsize=(10, 8))
sns.barplot(data=grouped_df, x='Gain', y='Display_Name', palette='magma')

plt.title('Feature Importance: Drivers of Nairobi PM2.5 Concentrations', fontsize=15, fontweight='bold')
plt.xlabel('Aggregated Information Gain (Model Contribution)', fontsize=12)
plt.ylabel('')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save for your report/presentation
plt.savefig('nairobi_grouped_importance_final.png', dpi=300)
print("Aggregated chart saved successfully!")