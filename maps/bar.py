import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 1. CONFIGURATION & COLUMN MAPPINGS
# =====================================================================
CSV_GRID_PATH = "/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_full_grid_stage2_2025.csv"
SUBCOUNTY_GEOJSON = "/mnt/e/Airquality/Datasets/Processed/Pointsources/nairobisubcounty.geojson"
POP_RASTER_UTM37S = "/mnt/e/Airquality/Datasets/Processed/Static/population_density_aligned.tif"
OUTPUT_BAR_PLOT = "nairobi_pwe_comparison.png"

LAT_COL, LON_COL = "lat", "lon"
PM_COL = "final_pm25_physical"
DATE_COL = "calc_date"
THRESHOLD = 15.0 # The methodology threshold for exceedance 

# =====================================================================
# 2. DATA LOADING & SPATIAL PROCESSING (Optimized)
# =====================================================================
print("Loading datasets and performing spatial integration...")
df_daily = pd.read_csv(CSV_GRID_PATH)
df_daily[DATE_COL] = pd.to_datetime(df_daily[DATE_COL])

# Optimize: Overlay only unique coordinates to save memory
unique_coords = df_daily[[LAT_COL, LON_COL]].drop_duplicates().reset_index(drop=True)
gdf_points = gpd.GeoDataFrame(unique_coords, 
    geometry=gpd.points_from_xy(unique_coords[LON_COL], unique_coords[LAT_COL]), 
    crs="EPSG:4326")

# Spatial Join with Subcounties [cite: 88, 203]
subcounties = gpd.read_file(SUBCOUNTY_GEOJSON)
subcounty_col = 'subcounty' if 'subcounty' in subcounties.columns else subcounties.columns[0]
gdf_points = gpd.sjoin(gdf_points, subcounties[[subcounty_col, 'geometry']], how='inner', predicate='intersects')

# Extract Population Weights using UTM Projection
gdf_points_utm = gdf_points.to_crs(epsg=32737)
coord_list = [(geom.x, geom.y) for geom in gdf_points_utm.geometry]

with rasterio.open(POP_RASTER_UTM37S) as src:
    pop_samples = [val[0] for val in src.sample(coord_list)]

gdf_points['pop_density'] = [max(0.0, float(v)) for v in pop_samples]

# Broadcast back to daily temporal data
df_merged = df_daily.merge(gdf_points[[LAT_COL, LON_COL, subcounty_col, 'pop_density']], on=[LAT_COL, LON_COL])

# =====================================================================
# 3. COMPUTE POPULATION-WEIGHTED EXPOSURE (PWE)
# =====================================================================
# PWE = Σ (Conc_i * Pop_i) / Σ Pop_i 
def calculate_pwe(group):
    weights = group['pop_density']
    if weights.sum() == 0: return np.nan
    return (group[PM_COL] * weights).sum() / weights.sum()

print("Calculating Annual PWE...")
annual_pwe = df_merged.groupby(subcounty_col).apply(calculate_pwe).reset_index(name='PWE')
annual_pwe = annual_pwe.sort_values(by='PWE', ascending=False).reset_index(drop=True)

# =====================================================================
# 4. PLOT HORIZONTAL BAR GRAPH (Publication Style)
# =====================================================================
print("Generating direct comparison visualization...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors based on threshold exceedance
colors = ['#d73027' if x > THRESHOLD else '#4575b4' for x in annual_pwe['PWE']]

bars = ax.barh(annual_pwe[subcounty_col], annual_pwe['PWE'], color=colors, edgecolor='black', alpha=0.8)

ax.invert_yaxis()
# Add the dotted red threshold line [Direct Comparison Request]
ax.axvline(x=THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold ({THRESHOLD} μg/m³)')

# Annotate values on bars
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
            va='center', fontsize=10, fontweight='bold')

# Formatting for Clarity [cite: 142, 719]
ax.set_title(f'Nairobi Subcounties: Annual Population-Weighted $PM_{{2.5}}$ Exposure', fontsize=16, pad=20)
ax.set_xlabel('Population-Weighted Mean ($\mu g/m^3$)', fontsize=12, fontweight='bold')
ax.set_ylabel('Subcounty', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', frameon=True, shadow=True)

plt.tight_layout()
plt.savefig(OUTPUT_BAR_PLOT, dpi=300)
print(f"Refined visualization saved to: {OUTPUT_BAR_PLOT}")