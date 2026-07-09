import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

# Try to import adjustText for drawing pointer lines on overlapping labels
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("\n💡 TIP: Run 'conda install -c conda-forge adjusttext' to draw pointer lines for labels.\n")

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURATION & COLUMN MAPPINGS
# =====================================================================
CSV_GRID_PATH = "/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_full_grid_stage2_2025.csv"
SUBCOUNTY_GEOJSON = "/mnt/e/Airquality/Datasets/Processed/Pointsources/nairobisubcounty.geojson"
POP_RASTER_UTM37S = "/mnt/e/Airquality/Datasets/Processed/Static/population_density_aligned.tif"

OUTPUT_CSV = "nairobi_true_annual_seasonal_pwe.csv"
OUTPUT_HEATMAP = "nairobi_pwe_matrix_heatmap.png"
OUTPUT_ANNUAL_MAP = "nairobi_annual_pwe_map.png"
OUTPUT_SEASONAL_MAP = "nairobi_seasonal_pwe_map.png"

LAT_COL, LON_COL = "lat", "lon"
PM_COL = "final_pm25_physical"
DATE_COL = "calc_date"

SEASON_ORDER = ['Hot and Dry (JF)', 'Long Rains (MAM)', 'Cold Dry (JJAS)', 'Short Rains (OND)']

def assign_custom_season(month):
    if month in [1, 2]: return 'Hot and Dry (JF)'
    elif month in [3, 4, 5]: return 'Long Rains (MAM)'
    elif month in [6, 7, 8, 9]: return 'Cold Dry (JJAS)'
    elif month in [10, 11, 12]: return 'Short Rains (OND)'
    return 'Unknown'

# =====================================================================
# 2. LOAD DATA & SPATIAL INTEGRATION
# =====================================================================
print("Loading daily PM2.5 CSV grid...")
df_daily = pd.read_csv(CSV_GRID_PATH)
df_daily[DATE_COL] = pd.to_datetime(df_daily[DATE_COL])
df_daily['month'] = df_daily[DATE_COL].dt.month
df_daily['season'] = df_daily['month'].apply(assign_custom_season)

# Isolate unique grid locations
unique_coords = df_daily[[LAT_COL, LON_COL]].drop_duplicates().reset_index(drop=True)
gdf_points = gpd.GeoDataFrame(
    unique_coords,
    geometry=gpd.points_from_xy(unique_coords[LON_COL], unique_coords[LAT_COL]),
    crs="EPSG:4326"
)

# Spatial join with subcounty boundaries
subcounties = gpd.read_file(SUBCOUNTY_GEOJSON)
if subcounties.crs is None or subcounties.crs.to_string() != "EPSG:4326":
    subcounties = subcounties.to_crs("EPSG:4326")

subcounty_col = 'subcounty' if 'subcounty' in subcounties.columns else subcounties.columns[0]
gdf_points = gpd.sjoin(gdf_points, subcounties[[subcounty_col, 'geometry']], how='inner', predicate='intersects')
gdf_points = gdf_points.drop(columns=['index_right'])

# Extract population weights via UTM Zone 37S
print("Extracting Population Density...")
gdf_points_utm = gdf_points.to_crs(epsg=32737)
coord_list = [(geom.x, geom.y) for geom in gdf_points_utm.geometry]

with rasterio.open(POP_RASTER_UTM37S) as src:
    pop_samples = [val[0] for val in src.sample(coord_list)]

gdf_points['pop_density'] = [max(0.0, float(x)) for x in pop_samples]

# Merge static spatial weights back to the primary daily table
df_merged = df_daily.merge(
    gdf_points[[LAT_COL, LON_COL, subcounty_col, 'pop_density']], 
    on=[LAT_COL, LON_COL], 
    how='inner'
)

# =====================================================================
# 3. COMPUTE TRUE POPULATION-WEIGHTED EXPOSURE (PWE)
# =====================================================================
print("Calculating True Annual and Seasonal PWE baselines...")
df_valid = df_merged.dropna(subset=['pop_density', PM_COL])

def calculate_pwe(group):
    weights = group['pop_density']
    if weights.sum() == 0: return np.nan
    return (group[PM_COL] * weights).sum() / weights.sum()

# Compute Baselines
annual_pwe = df_valid.groupby(subcounty_col).apply(calculate_pwe).reset_index(name='Annual')
seasonal_pwe = df_valid.groupby([subcounty_col, 'season']).apply(calculate_pwe).unstack()
seasonal_pwe = seasonal_pwe.reindex(columns=SEASON_ORDER)

# Compile finalized matrix
df_final = annual_pwe.merge(seasonal_pwe, on=subcounty_col, how='left')
df_final = df_final.sort_values(by='Annual', ascending=False).reset_index(drop=True)
df_final.to_csv(OUTPUT_CSV, index=False)

# Map PWE scores back to the lat/lon grid points so we can use the boxy ax.scatter method
df_grid_pwe = gdf_points[[LAT_COL, LON_COL, subcounty_col]].merge(df_final, on=subcounty_col, how='left')

# Generate Initials & Centroids for labels
def generate_clean_initials(name):
    name_str = str(name).strip().upper()
    for suffix in [" SC", " SUB-COUNTY", " SUBCOUNTY", " SUB COUNTY"]:
        if name_str.endswith(suffix): name_str = name_str.replace(suffix, "").strip()
    if "STAREHE" in name_str: return "ST"
    if "MAKADARA" in name_str: return "MK"
    if "DAGORETTI NORTH" in name_str: return "DN"
    if "DAGORETTI SOUTH" in name_str: return "DS"
    words = name_str.replace('-', ' ').split()
    return words[0][0] + words[1][0] if len(words) >= 2 else words[0][:2]

gdf_stats = subcounties.copy()
gdf_stats['initials'] = gdf_stats[subcounty_col].apply(generate_clean_initials)

# Global Min/Max for unified Colorbars across Maps
vmin = df_final[['Annual'] + SEASON_ORDER].min().min()
vmax = df_final[['Annual'] + SEASON_ORDER].max().max()

# Bounding box logic
minx, miny, maxx, maxy = subcounties.total_bounds
pad_x = (maxx - minx) * 0.015 
pad_y = (maxy - miny) * 0.015

# =====================================================================
# 4. VISUALIZATION 1: MATRIX HEATMAP
# =====================================================================
# =====================================================================
# 4. VISUALIZATION 1: MATRIX HEATMAP
# =====================================================================
print("Rendering Matrix Heatmap...")
heatmap_matrix = df_final.set_index(subcounty_col)[['Annual'] + SEASON_ORDER]

# 1. NEW: Clean the y-axis labels by removing the suffix
heatmap_matrix.index = heatmap_matrix.index.map(
    lambda x: str(x).upper().replace(" SUB-COUNTY", "").replace(" SUBCOUNTY", "").replace(" SUB COUNTY", "").replace(" SC", "").strip().title()
)

plt.figure(figsize=(16, 12)) 
sns.set_theme(style="white") 

ax = sns.heatmap(heatmap_matrix, annot=True, cmap='YlOrRd', fmt=".2f",
                 vmin=vmin, vmax=vmax, linewidths=0.5,
                 annot_kws={'size': 13, 'weight': 'bold'}) 

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=13)
cbar.set_label('Population-Weighted $PM_{2.5}$ ($\mu g/m^3$)', size=14, weight='bold')

plt.title('Nairobi Subcounties: Population-Weighted $PM_{2.5}$', fontsize=20, fontweight='bold', pad=25)
plt.ylabel('Subcounty', fontsize=15, fontweight='bold', labelpad=10)
plt.xlabel('Temporal Window', fontsize=15, fontweight='bold', labelpad=10)

# 2. INCREASED: Bumped y-ticks to 14 to fill the newly freed space
plt.xticks(rotation=0, fontsize=13, fontweight='bold') 
plt.yticks(rotation=0, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_HEATMAP, dpi=400, bbox_inches='tight') 
plt.close()

# =====================================================================
# 5. VISUALIZATION 2: ANNUAL PWE MAP (USING AX.SCATTER BOXES)
# =====================================================================
def plot_annual_map(data, title, filename):
    """Plots the single Annual map with labels, pointer lines, and a stats sidebar using ax.scatter."""
    print(f"Plotting {title}...")
    fig, ax = plt.subplots(figsize=(14, 10), dpi=400)

    # 1. Base Map & Scatter
    subcounties.plot(ax=ax, color='#1e1e1e')
    sc = ax.scatter(data['lon'], data['lat'], c=data['Annual'],
                    cmap='magma', vmin=vmin, vmax=vmax, s=180, marker='s', edgecolor='none')
    subcounties.boundary.plot(ax=ax, color='#888888', linewidth=1.2)

    # 2. Add Initials
    texts = []
    for idx, row in gdf_stats.iterrows():
        if pd.notna(row['initials']):
            centroid = row.geometry.centroid
            texts.append(ax.text(centroid.x, centroid.y, row['initials'], 
                                 fontsize=10, weight='bold', color='white',
                                 ha='center', va='center', 
                                 path_effects=[pe.withStroke(linewidth=1.5, foreground="black")]))

    # 3. Handle Overlaps with Pointer Lines
    if HAS_ADJUST_TEXT:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='white', lw=1.0, alpha=0.8),
                    expand_points=(1.3, 1.3), expand_text=(1.2, 1.2),
                    force_text=(0.5, 0.5), force_points=(0.2, 0.2))

    # 4. Generate Sidebar String
   # 4. Generate Sidebar String
    sidebar_str = "Annual PWE (µg/m³)\n" + "═" * 25 + "\n"
    for _, row in df_final.iterrows():
        raw_name = str(row[subcounty_col]).upper()
        clean_name = raw_name.replace(" SUB-COUNTY", "").replace(" SUBCOUNTY", "").replace(" SUB COUNTY", "").replace(" SC", "").strip().title()
        
        # Get matching initials for the sidebar
        initial = gdf_stats.loc[gdf_stats[subcounty_col] == row[subcounty_col], 'initials'].values[0]
        
        # Use .ljust(15) to keep numbers aligned
        sidebar_str += f"{initial} : {clean_name.ljust(15)} {row['Annual']:.1f}\n"

    # 5. Formatting & Layout
    ax.set_title(title, fontsize=18, weight='bold', pad=15)
    ax.set_xlabel('Longitude', fontsize=12, weight='bold')
    ax.set_ylabel('Latitude', fontsize=12, weight='bold')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect('equal')

    # Push the plot to the left to make room for the wider/taller legend
    fig.subplots_adjust(right=0.72)

    # Colorbar
    cbar_ax = fig.add_axes([0.74, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Population-Weighted $PM_{2.5} \, (\mu g/m^3)$', fontsize=14, weight='bold')
    cbar.ax.tick_params(labelsize=11)

    # ---------------------------------------------------------
    # NEW: Spaced out and padded Sidebar text box
    # ---------------------------------------------------------
    fig.text(0.82, 0.5, sidebar_str, 
             fontsize=12, 
             family='monospace',
             weight='semibold',         # <-- Makes text slightly bolder/crisper
             color='black', 
             va='center', 
             ha='left',
             linespacing=1.8,           # <-- INCREASES VERTICAL SPACE BETWEEN SUCOUNTIES
             bbox=dict(facecolor='#f8f9fa', edgecolor='gray', boxstyle='round,pad=1.5')) # <-- pad=1.5 MAKES THE BOX BIGGER
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

plot_annual_map(df_grid_pwe, 'Annual Population-Weighted Mean $PM_{2.5}$ & Ranking (2025)', OUTPUT_ANNUAL_MAP)

# =====================================================================
# 6. VISUALIZATION 3: SEASONAL PWE GRID (USING AX.SCATTER BOXES)
# =====================================================================
print("Rendering Seasonal PWE Grid...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True, dpi=300)

for i, season in enumerate(SEASON_ORDER):
    ax = axes.flatten()[i]
    
    # Base Map & Scatter
    subcounties.plot(ax=ax, color='#1e1e1e')
    sc = ax.scatter(df_grid_pwe['lon'], df_grid_pwe['lat'], c=df_grid_pwe[season],
                    cmap='magma', vmin=vmin, vmax=vmax, s=180, marker='s', edgecolor='none')
    subcounties.boundary.plot(ax=ax, color='#888888', linewidth=1.2)
    
    ax.set_title(season, fontsize=14, weight='bold')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_xlim(minx - pad_x, maxx + pad_x); ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect('equal')

fig.subplots_adjust(right=0.88, top=0.90, bottom=0.05, wspace=0.05, hspace=0.15)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_label('Population-Weighted $PM_{2.5} \, (\mu g/m^3)$', fontsize=14, weight='bold')

fig.supxlabel('Longitude', fontsize=14, weight='bold', y=0.01)
fig.supylabel('Latitude', fontsize=14, weight='bold', x=0.04)
plt.suptitle('Seasonal Population-Weighted $PM_{2.5}$ Exposure (2025)', fontsize=20, weight='bold', y=0.98)

plt.savefig(OUTPUT_SEASONAL_MAP, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Complete: CSV, Matrix Heatmap, and Scatter Grid Maps generated.")