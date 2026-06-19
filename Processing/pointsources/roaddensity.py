import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from shapely.geometry import box
from rasterio.features import rasterize

def generate_road_density_raster(template_raster, roads_geojson, boundary_geojson, output_path):
    print("🚀 Starting Road Density Generation...")
    
    # 1. Load Master Grid
    with rasterio.open(template_raster) as temp:
        target_crs = temp.crs
        transform = temp.transform
        width = temp.width
        height = temp.height
        meta = temp.meta.copy()

    # 2. Load and Reproject Data
    roads = gpd.read_file(roads_geojson).to_crs(target_crs)
    boundary = gpd.read_file(boundary_geojson).to_crs(target_crs)

    # 3. Create the Pixel Grid Polygons
    print(" -> Creating 1km grid polygons...")
    res = transform[0]  # Assuming 1000m
    grid_cells = []
    for r in range(height):
        for c in range(width):
            x, y = transform * (c, r)
            # Create a box for the pixel
            grid_cells.append(box(x, y - res, x + res, y))
    
    grid_gdf = gpd.GeoDataFrame({'pixel_id': range(len(grid_cells))}, 
                                geometry=grid_cells, crs=target_crs)

    # 4. Intersect Roads with Grid (The Math)
    print(" -> Intersecting roads (this may take a minute)...")
    # This cuts the roads at the pixel boundaries
    roads_intersected = gpd.overlay(roads, grid_gdf, how='intersection')
    
    # Calculate length in kilometers
    roads_intersected['len_km'] = roads_intersected.length / 1000.0
    
    # Sum length per pixel
    density_per_pixel = roads_intersected.groupby('pixel_id')['len_km'].sum()

    # 5. Map back to the Grid Array
    density_array = np.zeros(height * width, dtype='float32')
    for idx, val in density_per_pixel.items():
        density_array[idx] = val
    
    density_grid = density_array.reshape(height, width)

    # 6. Apply Boundary Mask (Clip to Nairobi)
    print(" -> Masking to Nairobi boundary...")
    mask = rasterize(shapes=boundary.geometry, out_shape=(height, width), 
                     transform=transform, fill=0, default_value=1, all_touched=True)
    
    density_grid[mask == 0] = np.nan

    # 7. Save to File
    meta.update({'dtype': 'float32', 'nodata': np.nan, 'count': 1})
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(density_grid, 1)
        dst.set_band_description(1, 'Road_Density_km_per_sqkm')

    print(f" Road Density saved to: {output_path}\n")

# ==========================================
# 2. CALL THE FUNCTION
# ==========================================

# Use the same paths as your previous script
MASTER_VIIRS = "/mnt/e/Airquality/Datasets/Interim/reprojected_viirs/VIIRS_Nairobi_UTM_A2025002_1036.tif"
NAIROBI_BOUNDARY = "/mnt/e/Airquality/Datasets/RAW/BOUNDARY/nairobi_Boundary.geojson"
ROADS_DATA = "/mnt/e/Airquality/Datasets/Processed/Pointsources/roads.geojson" # The Overpass Turbo export
OUTPUT_FILE = "/mnt/e/Airquality/Datasets/Processed/Static/road_density_aligned.tif"

generate_road_density_raster(
    template_raster=MASTER_VIIRS,
    roads_geojson=ROADS_DATA,
    boundary_geojson=NAIROBI_BOUNDARY,
    output_path=OUTPUT_FILE
)