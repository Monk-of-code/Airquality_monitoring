To use the Copernicus Data Store (CDS) API within WSL (Windows Subsystem for Linux), you must configure your credentials specifically for the Linux environment, as WSL does not automatically read your Windows user folder. 
## WSL Setup Guide for CDS API## 1. Configure Credentials
The cdsapi library looks for a hidden file named .cdsapirc in your WSL home directory (/home/wanji/). 
Option A: Copy from Windows (Recommended)
If you already created the file in Windows, run this in your WSL terminal:

cp /mnt/c/Users/wanji/.cdsapirc ~/.cdsapirc

Option B: Create manually in WSL

   1. Open the file editor: nano ~/.cdsapirc
   2. Paste your API details exactly as they appear on your [CDS Profile page](https://cds.climate.copernicus.eu/user-guide):
   
   url: https://cds.climate.copernicus.eu/api
   key: YOUR_PERSONAL_ACCESS_TOKEN
   
   3. Save and exit (Press Ctrl+O, Enter, then Ctrl+X). 

## 2. Install the Client
Inside your WSL virtual environment, ensure the library is installed: 

pip install cdsapi

## 3. Critical: Accept Dataset Licenses [1, 2, 7, 9] 
The API will fail if you haven't manually agreed to the terms for each specific dataset. Visit these pages, scroll to the bottom, and click Accept: [6, 10, 11] 

* ERA5-Land Terms
* ERA5 Single Levels (PBLH) Terms 

Here is a structured README.md that covers the environment setup, dependency installation, and the critical HDF4 driver verification we just performed.

# MAIAC AOD Data Fusion Pipeline
Got it — here’s a **clean README section focused only on Conda environment setup, management, and cleanup** (no project details).

---

````markdown
# ⚙️ Conda Environment Management Guide

This project uses Conda to manage all geospatial, machine learning, and scientific computing dependencies in a reproducible way.

---

# 🧱 1. Create a New Environment

Use a clean Python version (recommended: 3.11 for geospatial stability):

```bash
conda create -n airquality python=3.11 -y
conda activate airquality
````

---

# 📦 2. Install Packages (Recommended Order)

## Core geospatial stack (always use conda-forge)

```bash
conda install -c conda-forge \
gdal \
rasterio \
rioxarray \
xarray \
netcdf4 \
pyhdf \
numpy \
pandas \
scipy \
matplotlib \
scikit-learn -y
```

---

## Python utilities (use pip only for pure Python packages)

```bash
pip install \
cdsapi \
python-dotenv \
tqdm \
psycopg2-binary
```

---

# 🔍 3. Check Environment

## Verify Python environment

```bash
which python
python --version
```

## Verify GDAL

```bash
gdalinfo --version
```

## Verify Python GDAL bindings

```bash
python -c "from osgeo import gdal; print(gdal.VersionInfo())"
```

---

# 📋 4. List Environments

```bash
conda env list
```

---

# 🔁 5. Activate / Deactivate Environment

## Activate

```bash
conda activate airquality
```

## Deactivate

```bash
conda deactivate
```

---

# 🧹 6. Remove an Environment (Clean Reset)

To completely delete an environment:

```bash
conda remove -n airquality --all -y
```

---

# 🧼 7. Clean Conda Cache (optional but useful)

Removes unused packages and cached files:

```bash
conda clean --all -y
```

---

# 🧯 8. Fix Common Issues

## ❌ GDAL / libgdal errors

Always reinstall using conda-forge:

```bash
conda install -c conda-forge gdal -y
```

Do NOT use pip for GDAL.

---

## ❌ Broken environment / conflicts

Best fix:

```bash
conda remove -n airquality --all -y
conda create -n airquality python=3.11 -y
```

---

## ❌ Mixing environments (avoid this)

Do NOT combine:

* conda + venv
* pip-installed GDAL
* system GDAL + conda GDAL

---

# 📌 9. Best Practices

✔ Always use `conda-forge` for geospatial libraries
✔ Use Python 3.11 for stability
✔ Keep one project = one environment
✔ Avoid installing GDAL via pip
✔ Prefer rebuilding over patching broken envs

---

# 🚀 Quick Reset Workflow (Recommended)

If things break:

```bash
conda remove -n airquality --all -y
conda clean --all -y
conda create -n airquality python=3.11 -y
conda activate airquality
```

Below is a **clean, end-to-end setup documentation** you can reuse for your Air Quality project — from Miniconda installation to environment creation and package setup (including a reproducible workflow like `requirements.txt` / `environment.yml`).

---

# 📦 AirQuality Project Setup Guide (Miniconda + Environment Reproducibility)

## 1. 🧹 Clean Installation Strategy (Recommended)

Always install Miniconda in a **single standard location**:

```bash
/home/wanji/miniconda3
```

Avoid installing in project folders or system directories.

---

## 2. ⬇️ Download Miniconda

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

If permission issues occur:

```bash
chmod +x Miniconda3-latest-Linux-x86_64.sh
```

---

## 3. ⚙️ Install Miniconda

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

During installation:

* Accept license → `yes`
* Install path:

  ```
  /home/wanji/miniconda3
  ```
* Allow initialization → `yes`

---

## 4. 🔄 Initialize Conda Shell

If not automatically done:

```bash
~/miniconda3/bin/conda init bash
exec bash
```

Verify:

```bash
conda --version
```

---

## 5. 🧠 Base Environment Behavior (IMPORTANT)

### ❓ Where should base be activated?

* You do NOT need to activate `base` manually for projects
* Best practice:

### ✅ Work flow:

```bash
conda activate base   # optional (for conda management only)
```

BUT for projects:

```bash
conda activate airquality
```

👉 Always work inside your **project-specific environment**, NOT base.

---

## 6. 📁 Create Project Environment

### Recommended: Create from scratch

```bash
conda create -n airquality python=3.10 -y
conda activate airquality
```

---

## 7. 📦 Install Core Packages (Geo + ML + NetCDF)

```bash
conda install -c conda-forge \
gdal rasterio rioxarray xarray netcdf4 hdf4 libgdal-hdf4 \
numpy pandas scipy scikit-learn matplotlib \
pyhdf shapely geopandas -y
```

---

## 8. 🧪 Fix GDAL/HDF Issues (Critical for your case)

Always ensure GDAL + plugins match:

```bash
conda install -c conda-forge gdal libgdal-hdf4 hdf4 -y
```

Check:

```bash
gdalinfo --formats | grep HDF
```

---

## 9. 🌍 Environment Variables (important for HDF/NetCDF)

Add to `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export GDAL_DATA=$CONDA_PREFIX/share/gdal
export PROJ_LIB=$CONDA_PREFIX/share/proj
```

Then reload:

```bash
source ~/.bashrc
```

---

## 10. 📌 Reproducible Setup (VERY IMPORTANT)

Instead of `requirements.txt`, use:

# ✅ BEST PRACTICE: `environment.yml`

Create file:

```bash
nano environment.yml
```

### Example:

```yaml
name: airquality
channels:
  - conda-forge
dependencies:
  - python=3.10
  - gdal
  - rasterio
  - rioxarray
  - xarray
  - netcdf4
  - hdf4
  - libgdal-hdf4
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - pyhdf
  - geopandas
  - shapely
```

---

## 11. 🚀 Create Environment from File

```bash
conda env create -f environment.yml
```

Activate:

```bash
conda activate airquality
```

---

## 12. 📄 Alternative: pip requirements.txt (NOT recommended for geospatial)

Only use for Python-only packages:

```txt
numpy
pandas
scikit-learn
matplotlib
```

Install:

```bash
pip install -r requirements.txt
```

❌ Avoid using pip for GDAL/HDF/rasterio unless absolutely necessary.

---

## 13. 🧪 Verification Checklist

Run:

```bash
which conda
which python
gdalinfo --version
python -c "import rasterio, xarray; print('OK')"
```

---

## 14. 🧭 Correct Workflow Summary

### Every session:

```bash
conda activate airquality
cd ~/projects/airquality/Processing
```

### Never:

* Install in base for project work
* Install Miniconda in random folders
* Mix pip + conda for GDAL stack unless needed

---

## 15. 🧨 Your Previous Error (Why it happened)

You had:

```
libgdal.so.36: cannot open shared object file
HDF driver missing
wrong GDAL build mismatch
```

Cause:

* Mixed GDAL installs
* Partial plugin installation
* Missing conda-forge dependency alignment

Fix:
👉 Clean environment + conda-forge-only stack (above)

---

## If you want next step

I can also help you:

✔ rebuild your PM2.5 pipeline cleanly
✔ fix your MAIAC AOD ingestion (HDF parsing)
✔ or design a reproducible research pipeline for Nairobi 1km grid

Just tell me 👍
