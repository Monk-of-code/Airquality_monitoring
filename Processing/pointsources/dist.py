import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.spatial.distance import cdist
import os

def generate_and_clip_influence_raster(template_raster, vector_path, boundary_geojson, output_path, decay_radius_m, layer_name):
    """
    Generates a continuous influence raster and perfectly clips it to the 
    Nairobi GeoJSON boundary, ensuring border pixels are kept (all_touched=True).
    """
    print(f"Generating & Clipping {layer_name} Influence Map...")
    
    # 1. Load Master Grid Geometry (UTM Projected)
    with rasterio.open(template_raster) as temp:
        target_crs = temp.crs
        transform = temp.transform
        width = temp.width
        height = temp.height
        meta = temp.meta.copy()

    # 2. Load Vector Points and force alignment to the UTM raster CRS
    print(" -> Loading vector sources...")
    points_gdf = gpd.read_file(vector_path).to_crs(target_crs)
    
    # Extract robust point geometry (handles both Point and Polygon centroids safely)
    point_coords = np.column_stack((points_gdf.geometry.centroid.x, points_gdf.geometry.centroid.y))

    # 3. Create the Nairobi Boundary Mask (all_touched=True)
    print(" -> Building exact boundary mask...")
    boundary_gdf = gpd.read_file(boundary_geojson).to_crs(target_crs)
    
    nairobi_mask = rasterize(
        shapes=boundary_gdf.geometry.values,
        out_shape=(height, width),
        transform=transform,
        fill=0,            # Outside Nairobi = 0
        default_value=1,   # Inside Nairobi = 1
        dtype='uint8',
        all_touched=True   # CRITICAL: Keeps pixels touching the exact border line
    )

    # 4. Generate the 1km Grid Centroids in Cartesian UTM meters
    print(" -> Generating grid coordinates...")
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    grid_coords = np.column_stack((np.array(xs).flatten(), np.array(ys).flatten()))

    # 5. The Physics: Calculate Euclidean Distances & Apply Exponential Decay
    # Validated: Because inputs are in flat UTM meters, Euclidean math is perfectly accurate
    print(f" -> Calculating matrix math for {len(grid_coords)} pixels and {len(point_coords)} sources...")
    dist_matrix = cdist(grid_coords, point_coords, metric='euclidean')
    exp_matrix = np.exp(-(dist_matrix / decay_radius_m))
    cumulative_influence = np.sum(exp_matrix, axis=1)

    # 6. Reassemble and Apply the Polygon Mask
    print(" -> Applying mask and saving GeoTIFF...")
    influence_grid = cumulative_influence.reshape(height, width).astype('float32')
    
    # Instantly erase the bounding box edges outside the Nairobi polygon
    influence_grid[nairobi_mask == 0] = np.nan

    meta.update({
        'count': 1,
        'dtype': 'float32',
        'nodata': np.nan
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(influence_grid, 1)
        dst.set_band_description(1, f'{layer_name}_Influence')

    print(f"Saved perfectly clipped grid to {output_path}\n")

# =====================================================================
# EXECUTE THE PIPELINE GENERATION
# =====================================================================

MASTER_VIIRS = "/mnt/e/Airquality/Datasets/Interim/reprojected_viirs/VIIRS_Nairobi_UTM_A2025002_1036.tif"
NAIROBI_BOUNDARY = "/mnt/e/Airquality/Datasets/RAW/BOUNDARY/nairobi_Boundary.geojson"

# 1. Generate Industrial Influence Map
generate_and_clip_influence_raster(
    template_raster=MASTER_VIIRS,
    vector_path="/mnt/e/Airquality/Datasets/Processed/Pointsources/manmade_industries.geojson",
    boundary_geojson=NAIROBI_BOUNDARY,
    output_path="/mnt/e/Airquality/Datasets/Processed/Static/industry_influence_aligned.tif",
    decay_radius_m=2000.0,
    layer_name="Industry"
)

# 2. Generate Transit Station Influence Map
generate_and_clip_influence_raster(
    template_raster=MASTER_VIIRS,
    vector_path="/mnt/e/Airquality/Datasets/Processed/Pointsources/transitpoints.geojson",
    boundary_geojson=NAIROBI_BOUNDARY,
    output_path="/mnt/e/Airquality/Datasets/Processed/Static/transit_influence_aligned.tif",
    decay_radius_m=300.0,
    layer_name="Transit"
)

# 3. CRITICAL NEW FEATURE: Generate Waste & Landfill Influence Map
# Configured with an aggressive 3,000m decay radius to model continuous particulate dispersion
generate_and_clip_influence_raster(
    template_raster=MASTER_VIIRS,
    vector_path="/mnt/e/Airquality/Datasets/Processed/Pointsources/waste.geojson", # Ensure this path matches your file
    boundary_geojson=NAIROBI_BOUNDARY,
    output_path="/mnt/e/Airquality/Datasets/Processed/Static/waste_influence_aligned.tif",
    decay_radius_m=3000.0,
    layer_name="Waste"
)