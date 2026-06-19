# PM₂.₅ Estimation and Population Exposure Mapping — Nairobi County


## Overview

This project develops an **AOD-independent, two-stage XGBoost machine learning framework** to estimate daily ground-level PM₂.₅ concentrations across Nairobi County at 1 km × 1 km spatial resolution for the year 2025. It bypasses the severe satellite Aerosol Optical Depth (AOD) data gaps caused by Nairobi's equatorial cloud cover by instead using VIIRS Top-of-Atmosphere (TOA) reflectance, ERA5 meteorological reanalysis data, and an improved spatial autocorrelation feature (Pₛ, k=3).

**Key results:**
- Sample-based cross-validation R² = **0.7039**, RMSE = **4.61 µg/m³**
- Annual mean PM₂.₅ = **19.6 µg/m³** (exceeds WHO Interim Target 3 of 15 µg/m³)
- Highest population-weighted exposure: **Mathare (22.6 µg/m³)** and **Kamukunji (22.4 µg/m³)**

---


## Environment Setup

The entire pipeline runs in **WSL2 (Windows Subsystem for Linux 2)** inside an **Anaconda environment**. All packages are installed via the `-c conda-forge` channel to guarantee GDAL compatibility.

### 1. Install Anaconda inside WSL2

Open your WSL2 terminal and run:

```bash
# Download the Anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O anaconda.sh

# Run the installer
# Accept the licence, confirm the install location, and say yes to conda init
bash anaconda.sh

# Reload your shell so the conda command is available
source ~/.bashrc

# Verify
conda --version
```

> If you prefer a lighter install, use [Miniconda](https://docs.anaconda.com/miniconda/) instead — the environment steps below are identical.

---

### 2. Create and activate the conda environment

All packages must come from `-c conda-forge` so that GDAL's native C/C++ dependencies (PROJ, HDF4, HDF5, NetCDF) are resolved from the same build tree. Mixing channels (e.g., installing GDAL from `defaults` and rasterio from `conda-forge`) causes `CXXABI` / `libstdc++` ABI mismatches that silently break GDAL at import time.

```bash
# Create a fresh Python 3.11 environment
conda create -n airquality python=3.11 -c conda-forge -y

# Activate it — run this every time you open a new WSL2 terminal
conda activate airquality
```

You should see `(airquality)` prefixed to your shell prompt before proceeding.

---

### 3. Install dependencies

Install everything in a single `conda install` call so the solver can pin compatible versions across all packages simultaneously:

```bash
conda install -c conda-forge \
    gdal=3.12.3 \
    geopandas=1.1.3 \
    rasterio=1.4.4 \
    pyproj=3.7.2 \
    shapely=2.1.2 \
    rioxarray=0.19.0 \
    xarray=2026.4.0 \
    netcdf4=1.7.4 \
    h5netcdf=1.8.1 \
    hdf5=2.1.0 \
    hdf4=4.2.15 \
    numpy=2.4.3 \
    pandas=3.0.2 \
    scipy=1.17.1 \
    scikit-learn=1.8.0 \
    xgboost=3.2.0 \
    matplotlib=3.40.9 \
    seaborn=0.13.2 \
    statsmodels=0.14.6 \
    pyresample=1.35.0 \
    pykdtree=1.4.3 \
    postgresql=18.3 \
    psycopg2=2.9.11 \
    sqlalchemy=2.0.49 \
    requests=2.33.1 \
    aiohttp=3.13.5 \
    cupy=13.4.1 \
    python-dotenv \
    cdsapi \
    -y
```

Verify the critical installs before continuing:

```bash
python -c "from osgeo import gdal; print('GDAL OK:', gdal.__version__)"
python -c "import rasterio; print('rasterio OK:', rasterio.__version__)"
python -c "import xgboost as xgb; print('XGBoost OK:', xgb.__version__)"
python -c "import geopandas; print('GeoPandas OK:', geopandas.__version__)"
```

#### GPU / CUDA (optional but recommended)

XGBoost uses `device="cuda"` for RandomizedSearchCV and model training, and CuPy is used for GPU-accelerated array operations. Install the CUDA toolkit matching your driver:

```bash
conda install -c conda-forge cuda-toolkit=12.9 -y

# Verify
nvidia-smi
python -c "import cupy; print('CuPy OK:', cupy.__version__)"
```

---

### 4. Configure credentials

Copy the example config and fill in your API keys and database credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
AIRQO_API_TOKEN=your_airqo_token_here
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=airqouser
POSTGRES_PASSWORD=airqopassword
POSTGRES_DB=airqodb
```

The CDS API for ERA5 downloads uses a separate `~/.cdsapirc` file. Create it with:

```bash
cat > ~/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api
key: your-cds-api-key-here
EOF
```

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| Python | 3.11.15 | Core runtime |
| gdal | 3.12.3 | Geospatial I/O and raster format conversion |
| rasterio | 1.4.4 | Reading/writing raster datasets |
| geopandas | 1.1.3 | Vector data processing |
| pyproj | 3.7.2 | CRS transformations |
| shapely | 2.1.2 | Geometric operations |
| rioxarray | 0.19.0 | Raster analysis with xarray |
| xarray | 2026.4.0 | Multidimensional meteorological data |
| netcdf4 | 1.7.4 | ERA5 NetCDF reading |
| h5netcdf | 1.8.1 | NetCDF via HDF5 interface |
| hdf5 | 2.1.0 | HDF5 dataset reading (VIIRS) |
| hdf4 | 4.2.15 | HDF4 reading (MAIAC AOD) |
| numpy | 2.4.3 | Numerical array operations |
| pandas | 3.0.2 | Tabular data manipulation |
| scipy | 1.17.1 | Scientific computation, weight generation |
| scikit-learn | 1.8.0 | Cross-validation, hyperparameter search |
| xgboost | 3.2.0 | Stage 1 and Stage 2 PM₂.₅ models |
| matplotlib | 3.40.9 | Visualization |
| seaborn | 0.13.2 | Statistical plotting |
| statsmodels | 0.14.6 | Statistical analysis |
| pyresample | 1.35.0 | Satellite swath resampling |
| pykdtree | 1.4.3 | Fast KD-tree nearest-neighbour search |
| postgresql | 18.3 | PM₂.₅ sensor record storage |
| psycopg2 | 2.9.11 | Python–PostgreSQL driver |
| sqlalchemy | 2.0.49 | SQL query execution from Python |
| requests | 2.33.1 | Dataset downloads |
| aiohttp | 3.13.5 | Asynchronous API calls (AirQo, CDS) |
| cupy | 13.4.1 | GPU-accelerated array computation |
| QGIS | 3.44.9 | Spatial visualization and map production |
| GPU | NVIDIA RTX 5070 | Hardware acceleration for XGBoost and CV |

---

## Data Sources

| Dataset | Code / Product | Source | Native Resolution | Target |
|---|---|---|---|---|
| Ground PM₂.₅ (calibrated) | AirQo Network (76 sites) | AirQo API | Vector / WGS 84 | 1 km / EPSG:32737 |
| VIIRS TOA Reflectance | VNP02MOD | NASA Earthdata | 750 m / Cartesian | 1 km / EPSG:32737 |
| VIIRS Geolocation | VNP03MOD | NASA Earthdata | 750 m / Cartesian | N/A |
| VIIRS Cloud Mask | CLDMSK_L2_VIIRS_SNPP | NASA Earthdata | 750 m / Geodetic | N/A |
| MAIAC AOD | MCD19A2 | NASA Earthdata | 1 km / Cartesian | 1 km / EPSG:32737 |
| MODIS NDVI | MOD13A3 | NASA Earthdata | 1 km / Cartesian | 1 km / EPSG:32737 |
| ERA5-Land (surface met.) | — | ECMWF / CDS | 9 km | 1 km / EPSG:32737 |
| ERA5 (PBLH) | — | ECMWF / CDS | 31 km | 1 km / EPSG:32737 |
| Sentinel-5P TROPOMI | NO₂, SO₂, CO | ESA Copernicus | 5.5 km | 1 km / EPSG:32737 |
| Elevation | SRTM DEM | NASA / USGS | 30 m / EPSG:4326 | 1 km / EPSG:32737 |
| Building Heights | GHS-BUILT-H | EU JRC | 100 m / EPSG:4326 | 1 km / EPSG:32737 |
| Roads, Industry, Waste | OpenStreetMap | OSM | Vector / EPSG:4326 | 1 km / EPSG:32737 |
| Population Density | WorldPop | HDX (2019 census) | 1 km / EPSG:4326 | 1 km / EPSG:32737 |

All datasets are reprojected to **Arc 1960 / UTM Zone 37S (EPSG:32737)** and resampled to a uniform **1 km × 1 km** grid covering Nairobi County.

> **AOD note:** MAIAC AOD was evaluated but discarded. After strict QC filtering (clear sky + adjacency = 0), monthly spatial coverage dropped below 3% across all months (June, August, September, and November had 0% valid coverage). AOD could not supply sufficient training samples.

---

## Pipeline Overview

```
AirQo API ──► PostgreSQL ──────────────────────────────┐
                                                        │
NASA Earthdata (VIIRS VNP02MOD/VNP03MOD/CLDMSK) ──►   │
  TOA reflectance (M03, M05, M11) + geometry            │
  Cloud masking → SZA filter → KD-Tree resample ────►  │
                                                        │
ECMWF CDS (ERA5 / ERA5-Land) ──────────────────────►  │
  RH computation → dual aggregation → bilinear         │
  interpolation to 1 km ──────────────────────────►    │
                                                        ▼
Sentinel-5P TROPOMI (NO₂, SO₂, CO) ──────────────► Feature
OSM vectors (roads, industry, waste) ──────────────►  merge
SRTM DEM (30 m → median resample 1 km) ───────────►  table
GHS-BUILT-H (100 m → median resample 1 km) ───────►
WorldPop population density ───────────────────────►
3D temporal encodings (cyclic + linear DOY) ───────►
Exponential distance-weighted emission proxies ────►
Time-lagged meteorological features (1d/3d/7d) ────►
                                                        │
                                                        ▼
                                               Stage 1 XGBoost
                                          (reg:gamma objective)
                                          GPU RandomSearchCV 150 iter
                                          5-fold GroupKFold tuning
                                                        │
                                                        ▼
                                         Supplementary grid predictions
                                                        │
                                                        ▼
                                         Pₛ (k=3) spatial autocorrelation
                                         Haversine ball tree + Gaussian decay
                                                        │
                                                        ▼
                                               Stage 2 XGBoost
                                          (all Stage 1 features + Pₛ)
                                          10-fold CV (sample/site/time)
                                                        │
                                                        ▼
                                    Daily 1 km PM₂.₅ maps (Jan–Dec 2025)
                                                        │
                                          ┌─────────────┴──────────────┐
                                          ▼                            ▼
                               Hotspot analysis               Population-weighted
                               (WHO 15 µg/m³ threshold)       exposure (PWE)
```

---

## Scripts

Run scripts in numbered order. Each script writes intermediate outputs to `data/processed/` and logs to stdout.

### 01 — AirQo Ingestion

The AirQo ingestion phase is split into two separate sub-pipelines: **metadata ingestion** (grid and site discovery from the AirQo API) and **CSV PM₂.₅ ingestion** (loading the downloaded calibrated sensor readings into PostgreSQL). Both write to a `bronze` schema following a raw-landing pattern.

---

#### 1a. Metadata ingestion — `request.py / transform.py / load.py / main.py`

**What it does**

This pipeline discovers which AirQo monitoring grids and sensor sites exist in Nairobi and maps each site to its device IDs. It stores this metadata in three PostgreSQL tables that the rest of the pipeline uses to join sensor readings to spatial locations.

**Run it:**

```bash
cd scripts/airqo_metadata/
python main.py
```

**How it works, step by step:**

`request.py` handles all HTTP communication with the AirQo API (`https://api.airqo.net/api/v2`) using `aiohttp` for async requests. It implements:

- Paginated fetching of all grids via `/devices/grids/summary` — it keeps requesting pages (incrementing `skip`) until the `meta.total` field confirms all records have been received.
- Per-grid device discovery via `/devices/grids/{grid_id}/generate`, which returns the site IDs and device IDs associated with each grid.
- Exponential backoff retry logic (up to 4 attempts, with jitter) on HTTP errors `408, 429, 500, 502, 503, 504`.

`transform.py` converts the raw API JSON into flat pandas DataFrames:

- `grids_to_df()` — one row per grid, extracting `grid_id`, `name`, `admin_level`, `network`, and the full raw JSON as a `payload` column for auditability.
- `grids_summary_to_sites_df()` — one row per sensor site, extracting coordinates (`approximate_latitude`, `approximate_longitude`), administrative columns (`sub_county`, `county`, `region`), and site metadata.
- `grid_generate_to_site_devices_df()` — flattens the generate response into a `(grid_id, site_id, device_id)` mapping table. Handles three different response shapes the API may return.
- `filter_grids_by_name()` — filters to only Nairobi grids before running the more expensive per-grid generate calls.

`load.py` writes everything to PostgreSQL using SQLAlchemy + psycopg2:

- Creates the `bronze` schema and enables the `postgis` extension if not already present.
- Creates three tables: `airqo_metadata_grids`, `airqo_metadata_sites` (with a PostGIS `GEOMETRY(Point, 4326)` column generated from lat/lon), and `airqo_metadata_site_devices`.
- All inserts use `ON CONFLICT ... DO UPDATE` (upsert), so re-running the script safely refreshes stale metadata without duplicating rows.

`main.py` orchestrates the above: loads credentials from `.env`, runs the async pipeline with `asyncio.run()`, and prints row counts for each table.

**Tables created:**

```
bronze.airqo_metadata_grids       — one row per grid
bronze.airqo_metadata_sites       — one row per sensor site (with PostGIS geom)
bronze.airqo_site_devices         — site ↔ device mapping facts
```

---

#### 1b. CSV PM₂.₅ ingestion — `classifier.py / load.py / main.py`

**What it does**

The AirQo platform exports hourly sensor readings as monthly CSV files. This pipeline cleans those files and bulk-loads them into PostgreSQL, separating raw sensor readings from machine-learning-calibrated readings into distinct bronze tables.

**Expected folder structure on the E: drive (mounted in WSL2 as `/mnt/e/`):**

```
/mnt/e/Airquality/Datasets/RAW/CALIBRATEDPM2.5/
├── 2025_01/
│   ├── nairobi_sensors_jan_week1.csv
│   └── nairobi_sensors_jan_week2.csv
├── 2025_02/
│   └── ...
└── ...
```

**Run it:**

```bash
cd scripts/csv_ingestion/
python main.py
```

**How it works, step by step:**

`classifier.py` handles file-level cleaning before anything touches the database:

- Auto-detects the delimiter (CSV vs TSV) using Python's `csv.Sniffer` on the first 2 KB of each file — the AirQo export format varies.
- Classifies each file as either **raw** (contains `s1_pm2_5`, `s2_pm2_5` dual-sensor columns) or **calibrated** (contains `pm2_5_calibrated_value`) by inspecting column names. This determines which database table the file lands in.
- Normalises timestamps to `Africa/Nairobi` timezone (UTC+3) using `pandas.to_datetime` with `utc=True` followed by `tz_convert`.
- Coerces all numeric columns to float, replacing unparseable values with `NaN`.
- Fixes "Null Island" coordinates — rows where both `latitude == 0` and `longitude == 0` are set to `NULL` rather than being loaded with a false location in the Gulf of Guinea.
- Ensures schema alignment by adding any missing expected columns as `pd.NA` so the CSV always matches the target table structure regardless of export format version.
- Deduplicates on `(datetime, device_name)` before writing the cleaned file to a `_cleaned_data/` subfolder inside the month directory.

`load.py` handles database writes using a staging-table pattern to avoid type conflicts with PostgreSQL's strict type system:

1. Creates a temporary table mirroring the target bronze table structure (`LIKE bronze.{table_name} INCLUDING ALL`).
2. Drops the `geom` column from the temp table so `COPY` can work with the flat CSV (which has `latitude`/`longitude` columns, not a geometry object).
3. Bulk-loads the cleaned CSV into the temp table using `psycopg2`'s `copy_expert` with `COPY ... FROM STDIN`, which is significantly faster than row-by-row inserts.
4. Adds `source_file` and `batch_id` metadata columns back to the temp table and generates PostGIS geometry points with `ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)`.
5. Inserts from the temp table into the final bronze table using `ON CONFLICT (device_name, datetime) DO NOTHING`, making re-runs safe.

`main.py` iterates through all month subdirectories in sorted order, calling `clean_and_write()` then the appropriate loader for each file. Failed files are caught, logged, and skipped individually — the pipeline rolls back only the failing file and continues to the next.

**Tables written:**

```
bronze.airqo_raw          — raw dual-sensor readings (s1_pm2_5, s2_pm2_5)
bronze.airqo_calibrated   — ML-calibrated readings (pm2_5_calibrated_value)
```

Both tables include `source_file`, `batch_id`, `ingested_at` audit columns and a PostGIS `GEOMETRY(Point, 4326)` column derived from sensor coordinates.

---

#### 1c. ERA5 ingestion — `era5_ingest.py`

**What it does**

Downloads hourly ERA5-Land surface meteorological variables and ERA5 Planetary Boundary Layer Height (PBLH) for Nairobi County for the full year 2025 via the Copernicus CDS API.

**Run it:**

```bash
python scripts/era5_ingest.py
```

**How it works:**

The script uses Python's `ThreadPoolExecutor` with `MAX_WORKERS=3` to download all 12 months in parallel. For each month it submits two CDS API requests:

- **ERA5-Land** (`reanalysis-era5-land`): downloads `2m_temperature`, `2m_dewpoint_temperature`, `10m_u/v_wind_components`, `surface_pressure`, and `total_precipitation` at hourly frequency, clipped to the Nairobi bounding box (`[-1.17, 36.67, -1.46, 37.27]`), saved as a NetCDF file.
- **ERA5 PBLH** (`reanalysis-era5-single-levels`): downloads `boundary_layer_height` at the same temporal resolution and extent.

Each download uses a `download_with_retry()` wrapper that retries up to 3 times with exponential backoff (10 s, 20 s, 30 s) on failure. Files are skipped if they already exist on disk, so interrupted runs resume cleanly.

Output files land in:

```
/mnt/e/Airquality/Datasets/RAW/ERA5LAND/era5_land_2025_MM.nc
/mnt/e/Airquality/Datasets/RAW/ERA5PBLH/era5_pblh_2025_MM.nc
```

All download progress and errors are written to `/mnt/e/Airquality/Datasets/Logs/ingestionlogs/era5_ingestion.log` and mirrored to the terminal.

