import os
import re
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Try to import adjustText for drawing pointer lines on overlapping labels
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("\n💡 TIP: Run 'conda install -c conda-forge adjusttext' in your terminal. This allows the script to automatically draw pointer lines for overlapping subcounty labels on the Annual map.\n")

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURATION & SIZING
# =====================================================================
CSV_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_full_grid_stage2_2025.csv'
GEOJSON_PATH = '/mnt/e/Airquality/Datasets/Processed/Pointsources/nairobisubcounty.geojson' 
SUBCOUNTY_NAME_COL = 'subcounty'

# 75% of an A4 page area (A4 Landscape: 11.69 x 8.27 inches)
FIG_WIDTH = 11.69 * 0.866  # ~10.12 inches
FIG_HEIGHT = 8.27 * 0.866  # ~7.16 inches

# =====================================================================
# 2. DATA LOADING & PREPARATION
# =====================================================================
print("Loading data and computing spatial aggregations...")
df = pd.read_csv(CSV_PATH)
df['calc_date'] = pd.to_datetime(df['calc_date'])
df['month'] = df['calc_date'].dt.month

def get_season(m):
    if m in [1, 2]: return 'Hot_Dry (JF)'
    if m in [3, 4, 5]: return 'Long_Rains (MAM)'
    if m in [6, 7, 8, 9]: return 'Cold_Dry (JJAS)'
    if m in [10, 11, 12]: return 'Short_Rains (OND)'

df['season'] = df['month'].apply(get_season)

# Global min/max for a consistent colorbar across all maps
vmin = df['final_pm25_physical'].min()
vmax = df['final_pm25_physical'].max()

# Aggregations
annual_df = df.groupby(['lat', 'lon'])['final_pm25_physical'].mean().reset_index()
monthly_df = df.groupby(['month', 'lat', 'lon'])['final_pm25_physical'].mean().reset_index()

season_order = ['Hot_Dry (JF)', 'Long_Rains (MAM)', 'Cold_Dry (JJAS)', 'Short_Rains (OND)']
seasonal_df = df.groupby(['season', 'lat', 'lon'])['final_pm25_physical'].mean().reset_index()
seasonal_df['season'] = pd.Categorical(seasonal_df['season'], categories=season_order, ordered=True)

# =====================================================================
# 3. SPATIAL JOIN & LABEL GENERATION
# =====================================================================
gdf_sub = gpd.read_file(GEOJSON_PATH)
if gdf_sub.crs is None or gdf_sub.crs.to_string() != "EPSG:4326":
    gdf_sub = gdf_sub.to_crs("EPSG:4326")

grid_points = annual_df[['lat', 'lon']].drop_duplicates()
gdf_grid = gpd.GeoDataFrame(grid_points, geometry=gpd.points_from_xy(grid_points.lon, grid_points.lat), crs="EPSG:4326")

gdf_joined = gpd.sjoin(gdf_grid, gdf_sub, how="left", predicate="intersects")

# Calculate Mean PM2.5 per Subcounty for the Sidebar
annual_subcounty_mean = pd.merge(annual_df, gdf_joined[['lat', 'lon', SUBCOUNTY_NAME_COL]], on=['lat', 'lon'])
subcounty_pm25 = annual_subcounty_mean.groupby(SUBCOUNTY_NAME_COL)['final_pm25_physical'].mean().reset_index()

# Merge stats back with geometries
gdf_stats = gdf_sub.merge(subcounty_pm25, on=SUBCOUNTY_NAME_COL, how='left')

def generate_clean_initials(name):
    name_str = str(name).strip().upper()
    
    # Robustly strip out any variation of "Sub County" or "SC"
    name_str = re.sub(r'(?i)\s+(sub-?county|sc)\s*$', '', name_str)
            
    # Hardcoded overrides for common tricky names
    if "STAREHE" in name_str: return "ST"
    if "MAKADARA" in name_str: return "MK"
    if "DAGORETTI NORTH" in name_str: return "DN"
    if "DAGORETTI SOUTH" in name_str: return "DS"
    
    words = name_str.replace('-', ' ').split()
    if len(words) >= 2:
        return words[0][0] + words[1][0] 
    elif len(words) == 1:
        return words[0][:2] 
    return ""

gdf_stats['initials'] = gdf_stats[SUBCOUNTY_NAME_COL].apply(generate_clean_initials)

# Create sorted dataframe for the Sidebar ranking
sidebar_stats = gdf_stats.dropna(subset=['final_pm25_physical']).sort_values('final_pm25_physical', ascending=False)

# Bounding box logic for map bounds
minx, miny, maxx, maxy = gdf_sub.total_bounds
pad_x = (maxx - minx) * 0.015 
pad_y = (maxy - miny) * 0.015


# =====================================================================
# 4. PLOTTING FUNCTIONS
# =====================================================================

def plot_grid_maps(data, title, filename, layout=(1, 1), group_col=None, names=None):
    """Plots Seasonal and Monthly maps (Grid format, boundaries only, NO labels)"""
    print(f"Plotting {title}...")
    fig_width = 16 if layout[1] == 4 else 14
    fig_height = 9 if layout[1] == 4 else 10
    
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(fig_width, fig_height), sharex=True, sharey=True)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    unique_vals = names if names is not None else sorted(data[group_col].unique())
    
    for i, val in enumerate(unique_vals):
        search_val = i + 1 if (group_col == 'month' and names is not None) else val
        subset = data[data[group_col] == search_val]
        ax = axes[i]
        
        # Dark Base Map
        gdf_sub.plot(ax=ax, color='#1e1e1e') 
        
        if not subset.empty:
            sc = ax.scatter(subset['lon'], subset['lat'], c=subset['final_pm25_physical'],
                            cmap='magma', vmin=vmin, vmax=vmax, s=85, marker='s', edgecolor='none')
            
        # Draw subcounty boundaries ONLY (no text labels)
        gdf_sub.boundary.plot(ax=ax, color='#888888', linewidth=1.0)
        
        ax.set_title(val, fontsize=14, weight='bold')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect('equal')
    
    # Layout and Colorbar
    fig.subplots_adjust(right=0.88, top=0.90, bottom=0.08, wspace=0.05, hspace=0.15)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('$PM_{2.5} \, (\mu g/m^3)$', fontsize=14, weight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    fig.supxlabel('Longitude', fontsize=14, weight='bold', y=0.03)
    fig.supylabel('Latitude', fontsize=14, weight='bold', x=0.08)
    plt.suptitle(title, fontsize=20, weight='bold', y=0.98)
    
    # -------------------------------------------------------------
    # North Arrow, Stepped Scale Bar, and CRS for Grid Maps
    # -------------------------------------------------------------
    # North Arrow (Top Right)
    fig.text(0.97, 0.96, 'N\n▲', ha='center', va='top', fontsize=18, weight='bold')
    
    # CRS and Stepped Scale Bar (Bottom Left)
    scale_text = "CRS: EPSG:4326\n├─────┼─────┤\n0     5    10 km "
    fig.text(0.08, 0.02, scale_text, ha='left', va='bottom', fontsize=11, weight='bold')

    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black') 
    plt.close()


def plot_annual_map(data, title, filename):
    """Plots the single Annual map with labels, pointer lines, and a stats sidebar."""
    print(f"Plotting {title}...")
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)

    # 1. Base Map & Scatter
    gdf_sub.plot(ax=ax, color='#1e1e1e')
    sc = ax.scatter(data['lon'], data['lat'], c=data['final_pm25_physical'],
                    cmap='magma', vmin=vmin, vmax=vmax, s=180, marker='s', edgecolor='none')
    gdf_sub.boundary.plot(ax=ax, color='#888888', linewidth=1.2)

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
                    arrowprops=dict(arrowstyle='-', color='black', lw=1.0, alpha=0.8),
                    expand_points=(1.3, 1.3), expand_text=(1.2, 1.2),
                    force_text=(0.5, 0.5), force_points=(0.2, 0.2))

    # 4. Generate Sidebar String
    sidebar_str = "Annual PM2.5 (µg/m³)\n" + "═" * 22 + "\n\n"
    for _, row in sidebar_stats.iterrows():
        # Robust Regex cleanup to remove any variation of Sub County from the legend
        raw_name = str(row[SUBCOUNTY_NAME_COL])
        clean_name = re.sub(r'(?i)\s+(sub-?county|sc)\s*$', '', raw_name).strip().title()
        sidebar_str += f"{row['initials']} : {clean_name}\n      {row['final_pm25_physical']:.1f}\n\n"

    # 5. Formatting & Layout
    ax.set_title(title, fontsize=18, weight='bold', pad=15)
    ax.set_xlabel('Longitude', fontsize=12, weight='bold')
    ax.set_ylabel('Latitude', fontsize=12, weight='bold')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect('equal')

    # Shrink plot area to make room for colorbar and sidebar box on the right
    fig.subplots_adjust(right=0.75)

    # Colorbar
    cbar_ax = fig.add_axes([0.77, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('$PM_{2.5} \, (\mu g/m^3)$', fontsize=14, weight='bold')
    cbar.ax.tick_params(labelsize=11)

    # Add the Sidebar text box
    fig.text(0.85, 0.5, sidebar_str, fontsize=10, family='monospace', va='center', ha='left',
             bbox=dict(facecolor='#f8f9fa', edgecolor='gray', boxstyle='round,pad=1'))

    # -------------------------------------------------------------
    # North Arrow, Stepped Scale Bar, and CRS for Annual Map
    # -------------------------------------------------------------
    # North Arrow (Inside Frame, Top Right)
    ax.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.85),
                arrowprops=dict(facecolor='black', edgecolor='black', width=4, headwidth=10),
                ha='center', va='top', xycoords='axes fraction', 
                fontsize=18, weight='bold', color='white',
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # CRS and Stepped Scale Bar (Inside Frame, Bottom Left)
    metadata_text = "CRS: EPSG:4326\n├─────┼─────┤\n0     5    10 km"
    ax.text(0.03, 0.03, metadata_text, transform=ax.transAxes,
            fontsize=11, weight='bold', color='black', va='bottom', ha='left',
            family='monospace', # Monospace keeps the step alignment clean
            path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# =====================================================================
# 5. ARTIFACT GENERATION
# =====================================================================
# 1. Annual Map (Labels + Sidebar)
plot_annual_map(annual_df, 'Annual Mean $PM_{2.5}$ Distribution & Subcounty Ranking (2025)', 'nairobi_annual_pm25_2025.png')

# 2. Seasonal Map (Boundaries Only)
plot_grid_maps(seasonal_df, 'Seasonal Mean $PM_{2.5}$ Distribution (2025)', 'nairobi_seasonal_pm25_2025.png', 
               layout=(2, 2), group_col='season', names=season_order)

# 3. Monthly Map (Boundaries Only)
plot_grid_maps(monthly_df, 'Monthly Mean $PM_{2.5}$ Distribution (2025)', 'nairobi_monthly_pm25_2025.png', 
               layout=(3, 4), group_col='month', 
               names=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

print("✅ Process Complete: Annual, Seasonal, and Monthly Maps Generated.")