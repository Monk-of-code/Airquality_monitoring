import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

def clip_viirs_archive(viirs_folder, output_folder, boundary_geojson):
    """
    Applies the exact Nairobi GeoJSON mask to an entire folder of VIIRS TIFFs 
    without running any heavy reprojection math.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Grab all VIIRS files
    viirs_files = glob.glob(os.path.join(viirs_folder, "*.tif"))
    print(f"Found {len(viirs_files)} VIIRS files to clip.")
    
    if not viirs_files:
        print("No VIIRS files found. Check your input path!")
        return

    # 2. Use the first VIIRS file to define the grid geometry
    template_path = viirs_files[0]
    with rasterio.open(template_path) as temp:
        master_transform = temp.transform
        master_width = temp.width
        master_height = temp.height
        master_crs = temp.crs
        
    # 3. Load the GeoJSON and create the Binary Mask ONCE
    print(" -> Building the exact Nairobi boundary mask...")
    gdf = gpd.read_file(boundary_geojson).to_crs(master_crs)
    
    nairobi_mask = rasterize(
        shapes=gdf.geometry.values,
        out_shape=(master_height, master_width),
        transform=master_transform,
        fill=0,            # Outside Nairobi = 0
        default_value=1,   # Inside Nairobi = 1
        dtype='uint8',
        all_touched=True   # Must match your other layers!
    )

    # 4. Loop through and "Cookie Cut" every VIIRS file
    for file_path in viirs_files:
        filename = os.path.basename(file_path)
        out_path = os.path.join(output_folder, f"clipped_{filename}")
        
        with rasterio.open(file_path) as src:
            meta = src.meta.copy()
            # Read all bands (VIIRS M03, M04, etc.)
            data = src.read()
            
            # Instantly erase everything outside the boundary across all bands!
            # We loop through the bands in case your VIIRS tiffs are multi-band
            for band_idx in range(data.shape[0]):
                data[band_idx][nairobi_mask == 0] = np.nan
                
            meta.update({'nodata': np.nan})
            
            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(data)
                
                # ---> THE FIX GOES EXACTLY HERE <---
                # Copy the original band names to the new clipped file
                if any(src.descriptions):
                    dst.descriptions = src.descriptions
                # -----------------------------------
                
        print(f" Clipped: {filename}")

    print("\n VIIRS Archive Successfully Clipped ")

# ==========================================
# EXECUTE THE CLIPPING
# ==========================================

VIIRS_INPUT = "/mnt/e/Airquality/Datasets/Interim/reprojected_viirs"
VIIRS_OUTPUT = "/mnt/e/Airquality/Datasets/Processed/VIIRS_Final_Clipped"
BOUNDARY = "/mnt/e/Airquality/Datasets/RAW/BOUNDARY/nairobi_Boundary.geojson"

if __name__ == "__main__":
    clip_viirs_archive(VIIRS_INPUT, VIIRS_OUTPUT, BOUNDARY)