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

# NEW: Strip all variations of the "Sub County" suffix to clean the labels
annual_pwe[subcounty_col] = annual_pwe[subcounty_col].apply(
    lambda x: str(x).upper().replace(" SUB-COUNTY", "").replace(" SUBCOUNTY", "").replace(" SUB COUNTY", "").replace(" SC", "").strip().title()
)

annual_pwe = annual_pwe.sort_values(by='PWE', ascending=False).reset_index(drop=True)

# =====================================================================
# 4. PLOT HORIZONTAL BAR GRAPH (Publication Style)
# =====================================================================
print("Generating direct comparison visualization...")

# Drop seaborn styles and use the clean default so we can control everything
plt.style.use('default') 
fig, ax = plt.subplots(figsize=(14, 10))

# 1. RESTORE AXES & GRID: Give the plot area a very subtle background color
ax.set_facecolor('#f8f9fa') 

# Draw strong, visible grid lines. 
# zorder=0 pushes the grid to the very back so it doesn't overlap your bars
ax.grid(True, axis='x', color='black', linestyle=':', alpha=0.4, zorder=0)
ax.grid(True, axis='y', color='black', linestyle=':', alpha=0.15, zorder=0)

# Define colors based on threshold exceedance
colors = ['#d73027' if x > THRESHOLD else '#4575b4' for x in annual_pwe['PWE']]

# 2. PLOT BARS: zorder=3 ensures the bars sit cleanly on top of the background grid
bars = ax.barh(annual_pwe[subcounty_col], annual_pwe['PWE'], 
               color=colors, edgecolor='black', alpha=0.9, height=0.7, zorder=3)

ax.invert_yaxis()

# Add the dotted red threshold line (made slightly thicker)
ax.axvline(x=THRESHOLD, color='red', linestyle='--', linewidth=3, zorder=4, 
           label=f'Threshold ({THRESHOLD} $\mu g/m^3$)')

# 3. FIX LEGIBILITY: Annotate values on bars with much larger, bolder text
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
            va='center', fontsize=14, fontweight='bold', color='black') # Bumped to 14

# 4. FIX FONT SIZES: Massive bump to Titles and Labels for 400 DPI legibility
ax.set_title('Nairobi Subcounties: Annual Population-Weighted $PM_{2.5}$ Exposure', 
             fontsize=22, fontweight='bold', pad=25)
ax.set_xlabel('Population-Weighted Mean ($\mu g/m^3$)', fontsize=16, fontweight='bold', labelpad=15)
ax.set_ylabel('Subcounty', fontsize=16, fontweight='bold', labelpad=15)

# Bump Tick Label Size
ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='x', labelsize=14)

# 5. RESTORE SPINES: Explicitly draw the bounding box so the graph feels grounded
# 5. ADJUST BOUNDARIES & SPINES
# Extend the x-axis by 10% to give the text labels plenty of breathing room
max_pwe = annual_pwe['PWE'].max()
ax.set_xlim(0, max_pwe * 1.1)

# Make the bounding box feel less restrictive
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_linewidth(1.5)
    
    if spine in ['top', 'right']:
        # Make the top and right edges a very faint gray (or set visible to False)
        ax.spines[spine].set_color('#cccccc') 
        ax.spines[spine].set_zorder(0) # Pushes it behind the text
    else:
        # Keep the bottom and left axes strong and black
        ax.spines[spine].set_color('black')

# Format the legend to be larger and clearly boxed
ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=14, 
          facecolor='white', edgecolor='black')

plt.tight_layout()

# Save with a white facecolor so transparent backgrounds don't cause issues in dark mode viewers
plt.savefig(OUTPUT_BAR_PLOT, dpi=400, bbox_inches='tight', facecolor='white')
print(f"Refined visualization saved to: {OUTPUT_BAR_PLOT}")