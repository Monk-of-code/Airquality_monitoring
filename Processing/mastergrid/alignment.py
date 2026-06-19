import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize

def align_and_clip_raster(input_path, output_path, dst_crs, dst_transform, dst_width, dst_height, master_meta, resampling_method, nairobi_mask):
    """Warps the raster to the Master Grid and applies the exact Nairobi polygon mask."""
    print(f" -> Warping & Clipping: {os.path.basename(input_path)}")
    
    with rasterio.open(input_path) as src:
        num_bands = src.count
        
        # Detect native NoData (ERA5 uses -9999.0)
        src_nodata = src.nodata if src.nodata is not None else np.nan
        
        # Create empty 3D array: (bands, height, width)
        destination_array = np.empty((num_bands, dst_height, dst_width), dtype=np.float32)

        # 1. Warp each band to the Master Grid
        for band_idx in range(1, num_bands + 1):
            reproject(
                source=rasterio.band(src, band_idx),
                destination=destination_array[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata, 
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=resampling_method
            )
            
            # 2. APPLY THE NAIROBI GEOJSON MASK
            # The mask is 1 inside Nairobi, 0 outside. We force all 0s to NaN.
            destination_array[band_idx - 1][nairobi_mask == 0] = np.nan

        # Setup output metadata
        out_meta = master_meta.copy()
        out_meta.update({
            'count': num_bands,
            'dtype': 'float32',
            'nodata': np.nan # The empty space is officially declared as NaN
        })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the perfectly aligned and clipped file
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(destination_array)
            if any(src.descriptions):
                dst.descriptions = src.descriptions

def run_universal_pipeline(tasks, template_raster, boundary_geojson):
    """Smart router that loads the Master Grid and Boundary Mask ONCE, then processes everything."""
    print(f"Loading Master Template: {os.path.basename(template_raster)}")
    
    # 1. Steal the Master Template's exact geometry
    with rasterio.open(template_raster) as temp:
        dst_crs = temp.crs
        dst_transform = temp.transform
        dst_width = temp.width
        dst_height = temp.height
        master_meta = temp.meta.copy()

    print(f"Loading Boundary Mask: {os.path.basename(boundary_geojson)}")
    # 2. Load the GeoJSON and ensure it shares the Master CRS (UTM 37S)
    gdf = gpd.read_file(boundary_geojson).to_crs(dst_crs)
    
    # 3. Create the Binary Mask Matrix
    # all_touched=True ensures edge pixels aren't deleted!
    nairobi_mask = rasterize(
        shapes=gdf.geometry.values,
        out_shape=(dst_height, dst_width),
        transform=dst_transform,
        fill=0,            # Outside Nairobi = 0
        default_value=1,   # Inside Nairobi = 1
        dtype='uint8',
        all_touched=True   # User required: Keep pixels touching the border
    )

    # 4. Loop through the configured tasks
    for task in tasks:
        task_name = task['name']
        input_target = task['input']
        output_target = task['output']
        resampling = task['resampling']
        
        print(f"\n[{task_name}] | Resampling: {resampling.name.upper()}")

        # FOLDER PROCESSING
        if os.path.isdir(input_target):
            tiff_files = glob.glob(os.path.join(input_target, "*.tif"))
            print(f"Found {len(tiff_files)} files in directory.")
            for input_file in tiff_files:
                output_file = os.path.join(output_target, f"aligned_{os.path.basename(input_file)}")
                align_and_clip_raster(
                    input_file, output_file, dst_crs, dst_transform, 
                    dst_width, dst_height, master_meta, resampling, nairobi_mask
                )

        # SINGLE FILE PROCESSING
        elif os.path.isfile(input_target):
            align_and_clip_raster(
                input_target, output_target, dst_crs, dst_transform, 
                dst_width, dst_height, master_meta, resampling, nairobi_mask
            )
        else:
            print(f" ERROR: Could not find input path -> {input_target}")

    print("\n UNIVERSAL PIPELINE COMPLETE! All grids are snapped and clipped to Nairobi.")


# ==========================================
# CONFIGURATION
# ==========================================
MASTER_VIIRS = "/mnt/e/Airquality/Datasets/Interim/reprojected_viirs/VIIRS_Nairobi_UTM_A2025002_1036.tif"
NAIROBI_BOUNDARY = "/mnt/e/Airquality/Datasets/RAW/BOUNDARY/nairobi_Boundary.geojson"

alignment_tasks = [
    {
        "name": "NDVI Monthly Stack",
        "input": "/mnt/e/Airquality/Datasets/Interim/Static/Nairobi_NDVI", 
        "output": "/mnt/e/Airquality/Datasets/Processed/Static/NDVI_A",
        "resampling": Resampling.bilinear
    },
    {
        "name": "ERA5 Daily Stack",
        "input": "/mnt/e/Airquality/Datasets/PROCESSED/DAILY_1KM_MET_STACK", 
        "output": "/mnt/e/Airquality/Datasets/Processed/ERA5_a",
        "resampling": Resampling.bilinear
    },
    {
        "name": "Building Heights",
        "input": "/mnt/e/Airquality/Datasets/Interim/Static/buildingheight/building_heights.tif",
        "output": "/mnt/e/Airquality/Datasets/Processed/Static/building_heights_aligned.tif",
        "resampling": Resampling.nearest # Nearest prevents blurring the ee.Reducer.median() values
    },
    {
        "name": "Elevation",
        "input": "/mnt/e/Airquality/Datasets/Interim/Static/Elevation/dem.tif",
        "output": "/mnt/e/Airquality/Datasets/Processed/Static/elevation.tif",
        "resampling": Resampling.nearest # Nearest preserves the exact ee.Reducer.median() bins
    },
    {
        "name": "Population Density",
        "input": "/mnt/e/Airquality/Datasets/Interim/Static/population_density.tif",
        # Updated to match your new Processed/Static directory structure!
        "output": "/mnt/e/Airquality/Datasets/Processed/Static/population_density_aligned.tif",
        "resampling": Resampling.bilinear # Population density is a continuous spatial gradient
    },
    {
        "name": "SO2_umol_m2",
        "input": "/mnt/e/Airquality/Datasets/Interim/gases/Nairobi_2025_SO2_umol_m2.tif",
        "output": "/mnt/e/Airquality/Datasets/Processed/gases/SO2_umol_m2.tif",
        "resampling": Resampling.bilinear
    },
    {
        "name": "NO2_umol_m2",
        "input": "/mnt/e/Airquality/Datasets/Interim/gases/Nairobi_2025_NO2_umol_m2.tif",
        "output": "/mnt/e/Airquality/Datasets/Processed/gases/NO2_umol_m2.tif",
        "resampling": Resampling.bilinear
    },
    {
        "name": "CO_mmol_m2",
        "input": "/mnt/e/Airquality/Datasets/Interim/gases/Nairobi_2025_CO_mmol_m2.tif",
        "output": "/mnt/e/Airquality/Datasets/Processed/gases/CO_mmol_m2.tif",
        "resampling": Resampling.bilinear
    }
]

if __name__ == "__main__":
    run_universal_pipeline(alignment_tasks, MASTER_VIIRS, NAIROBI_BOUNDARY)