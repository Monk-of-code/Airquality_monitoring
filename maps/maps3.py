import os
import re
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter

# Try to import adjustText for drawing pointer lines on overlapping labels
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("\n💡 TIP: Run 'conda install -c conda-forge adjusttext' in your terminal. "
          "This allows the script to automatically draw pointer lines for overlapping subcounty labels on the Annual map.\n")

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
CSV_PATH = '/mnt/e/Airquality/Datasets/Processed/pm2.5/nairobi_full_grid_stage2_2025.csv'
GEOJSON_PATH = '/mnt/e/Airquality/Datasets/Processed/Pointsources/nairobisubcounty.geojson' 
SUBCOUNTY_NAME_COL = 'subcounty'
OUTPUT_DIR = './plots'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# 2. DATA LOADING & PREPARATION
# =====================================================================
print("⏳ Loading data and computing spatial aggregations...")
df = pd.read_csv(CSV_PATH)
df['calc_date'] = pd.to_datetime(df['calc_date'])
df['month'] = df['calc_date'].dt.month

def get_season(m):
    if m in (1, 2):
        return 'Hot_Dry (JF)'
    elif m in (3, 4, 5):
        return 'Long_Rains (MAM)'
    elif m in (6, 7, 8, 9):
        return 'Cold_Dry (JJAS)'
    else:
        return 'Short_Rains (OND)'

df['season'] = df['month'].apply(get_season)

# Global min/max for a consistent colorbar scale across all maps
vmin = max(0.1, df['final_pm25_physical'].min()) # Avoid 0 or negative values for log scale
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

# Calculate Mean PM2.5 per Subcounty for potential layout enhancements
annual_subcounty_mean = pd.merge(annual_df, gdf_joined[['lat', 'lon', SUBCOUNTY_NAME_COL]], on=['lat', 'lon'])
subcounty_pm25 = annual_subcounty_mean.groupby(SUBCOUNTY_NAME_COL)['final_pm25_physical'].mean().reset_index()
gdf_stats = gdf_sub.merge(subcounty_pm25, on=SUBCOUNTY_NAME_COL, how='left')

def generate_clean_initials(name):
    name_str = str(name).strip().upper()
    name_str = re.sub(r'(?i)\s+(sub-?county|sc)\s*$', '', name_str)
            
    if "STAREHE" in name_str: return "ST"
    if "MAKADARA" in name_str: return "MK"
    if "DAGORETTI NORTH" in name_str: return "DN"
    if "DAGORETTI SOUTH" in name_str: return "DS"
    
    words = name_str.replace('-', ' ').split()
    if len(words) >= 2: return words[0][0] + words[1][0] 
    elif len(words) == 1: return words[0][:2] 
    return ""

gdf_stats['initials'] = gdf_stats[SUBCOUNTY_NAME_COL].apply(generate_clean_initials)
sidebar_stats = gdf_stats.dropna(subset=['final_pm25_physical']).sort_values('final_pm25_physical', ascending=False)

minx, miny, maxx, maxy = gdf_sub.total_bounds
pad_x = (maxx - minx) * 0.015 
pad_y = (maxy - miny) * 0.015

# =====================================================================
# 4. PLOTTING FUNCTIONS
# =====================================================================
# =====================================================================
# 4. PLOTTING FUNCTIONS
# =====================================================================

def plot_grid_maps(data, title, filename, layout=(3, 4), group_col='month', names=None):
    """
    Plots multi-panel maps with a shared horizontal colorbar, high-contrast inferno 
    colormap, logarithmic scaling, and centralized cartographic metadata.
    """
    print(f"🎨 Plotting grid layout for: {title}...")
    
    # FIX: Explicitly extract rows and columns from the layout tuple
    rows, cols = layout
    fig_width = cols * 4.5
    fig_height = rows * 3.8 + 2.5 
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    unique_vals = names if names is not None else sorted(data[group_col].unique())
    
    metadata_text = (
        "CARTOGRAPHIC METADATA\n"
        "─────────────────────\n"
        "• Projection: Geographic\n"
        "• Datum: WGS 84\n"
        "• EPSG: 4326\n"
        f"• Data Limits: {vmin:.1f} to {vmax:.1f} µg/m³"
    )
    
    metadata_placed = False
    sc = None # Placeholder for grid plot reference

    for i, ax in enumerate(axes):
        if i < len(unique_vals):
            val = unique_vals[i]
            search_val = i + 1 if (group_col == 'month' and names is not None) else val
            subset = data[data[group_col] == search_val]
            
            # Underlying muted base background 
            gdf_sub.plot(ax=ax, color='#252525', zorder=1) 
            
            # LogNorm scaling and high-contrast 'inferno' map
            if not subset.empty:
                sc = ax.scatter(
                    subset['lon'], subset['lat'], c=subset['final_pm25_physical'],
                    cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax), 
                    s=65, marker='s', edgecolor='none', zorder=2
                )
                
            # Sharp thin boundaries for crisp delineation
            gdf_sub.boundary.plot(ax=ax, color='#ffffff', linewidth=0.5, alpha=0.6, zorder=3)
            
            # Header Styling
            ax.set_title(str(val).upper(), fontsize=14, weight='bold', pad=10)
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.grid(True, linestyle=':', alpha=0.3, color='#cccccc', zorder=0)
            ax.set_xlim(minx - pad_x, maxx + pad_x)
            ax.set_ylim(miny - pad_y, maxy + pad_y)
            ax.set_aspect('equal')
        else:
            # Hide layout panels that exceed the number of data segments
            ax.axis('off')
            if not metadata_placed:
                ax.text(0.5, 0.5, metadata_text, transform=ax.transAxes,
                        fontsize=11, va='center', ha='center', family='monospace',
                        bbox=dict(facecolor='#f8f9fa', edgecolor='#bbbbbb', boxstyle='square,pad=1'))
                metadata_placed = True

    # If layout is perfectly filled (like 3x4), anchor metadata safely on the right margin
    if not metadata_placed:
        fig.subplots_adjust(right=0.83)
        fig.text(0.85, 0.5, metadata_text, fontsize=11, va='center', ha='left', family='monospace',
                 bbox=dict(facecolor='#f8f9fa', edgecolor='#bbbbbb', boxstyle='square,pad=1'))
    else:
        fig.subplots_adjust(right=0.95)

    # Master Shared Structural Labels
    fig.text(0.5, 0.16, 'Longitude', ha='center', va='center', fontsize=14, weight='bold')
    fig.text(0.03, 0.5, 'Latitude', ha='center', va='center', rotation='vertical', fontsize=14, weight='bold')

    # Unified Singular Scale Indicator instead of 12 duplicates
    fig.text(0.78, 0.09, 'Scale: 0 ── 5 ── 10 km\nOrientation: ⬆ N', 
             fontsize=10, weight='bold', family='monospace',
             bbox=dict(facecolor='white', edgecolor='#cccccc', boxstyle='round,pad=0.5'))

    # Fine layout padding control
    fig.subplots_adjust(bottom=0.22, top=0.91, left=0.08, wspace=0.04, hspace=0.18)
    
    # -------------------------------------------------------------
    # Shared Logarithmic Colorbar Execution
    # -------------------------------------------------------------
    if sc is not None:
        cbar_ax = fig.add_axes([0.25, 0.09, 0.45, 0.022]) 
        cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal', format=ScalarFormatter())
        cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
        cbar.set_label(r'Mean $PM_{2.5}$ Intensity ($\mu g/m^3$)', fontsize=13, weight='bold', labelpad=8)
        cbar.ax.tick_params(labelsize=10)

    # Figure Master Title
    fig.suptitle(title, fontsize=18, weight='bold', y=0.96)
    
    # Save step with extreme vector flattening to keep file sizing crisp
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot successfully saved to: {out_path}\n")
    plt.close()

# =====================================================================
# 5. EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # 1. Execute full Monthly 3x4 map Matrix
    plot_grid_maps(
        data=monthly_df, 
        title="Monthly Mean $PM_{2.5}$ Distribution (2025)", 
        filename="monthly_pm25_distribution.png", 
        layout=(4, 3), 
        group_col='month', 
        names=month_names
    )
    
    # 2. Execute full Seasonal 2x2 map Matrix
    plot_grid_maps(
        data=seasonal_df, 
        title="Seasonal Mean $PM_{2.5}$ Distribution (2025)", 
        filename="seasonal_pm25_distribution.png", 
        layout=(2, 2), 
        group_col='season', 
        names=season_order
    )
