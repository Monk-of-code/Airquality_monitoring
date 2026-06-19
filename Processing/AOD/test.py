import os
import subprocess
from pyhdf.SD import SD, SDC
from osgeo import gdal

def ensure_wsl_mount(drive_letter="E", mount_point="/mnt/e"):
    """Mounts a Windows drive in WSL if not already detected."""
    if not os.path.ismount(mount_point):
        print(f"--- Drive {mount_point} not detected. Attempting to mount {drive_letter}: ---")
        try:
            # Create mount point directory
            subprocess.run(['sudo', 'mkdir', '-p', mount_point], check=True)
            # Mount with drvfs and metadata enabled for Linux permissions
            subprocess.run([
                'sudo', 'mount', '-t', 'drvfs', f'{drive_letter}:', mount_point,
                '-o', f'uid={os.getuid()},gid={os.getgid()},metadata'
            ], check=True)
            print(f"✅ Successfully mounted {drive_letter}: to {mount_point}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Mount failed: {e}. Please ensure you have sudo privileges.")
            return False
    else:
        print(f"✅ Drive {mount_point} is already mounted.")
    return True

def diagnostic_test(file_path):
    print(f"\n--- Diagnostic Reader Test for: {os.path.basename(file_path)} ---")
    
    # 1. pyhdf Test
    print("\n[TEST 1: pyhdf]")
    try:
        hdf = SD(file_path, SDC.READ)
        datasets = hdf.datasets()
        print(f"✅ pyhdf Success! Found {len(datasets)} datasets.")
        # Identify the 0.55 AOD SDS
        sds_name = [s for s in datasets.keys() if "Optical_Depth_055" in s]
        if sds_name:
            print(f" - Found Target SDS: {sds_name[0]}")
        hdf.end()
    except Exception as e:
        print(f"❌ pyhdf Failed: {e}")

    # 2. GDAL Test
    print("\n[TEST 2: GDAL]")
    try:
        gdal.UseExceptions()
        ds = gdal.Open(file_path)
        if ds:
            sds = ds.GetSubDatasets()
            print(f"✅ GDAL Success! Found {len(sds)} subdatasets.")
            # Check for the 0.55 AOD path
            target_sds = [s[0] for s in sds if "Optical_Depth_055" in s[0]]
            if target_sds:
                print(f" - Found Target Subdataset: {target_sds[0][:60]}...")
        ds = None
    except Exception as e:
        print(f"❌ GDAL Failed: {e}")

if __name__ == "__main__":
    # 1. Ensure the E: drive is mounted
    if ensure_wsl_mount("E", "/mnt/e"):
        
        # 2. Define the exact path to your test file on the mounted drive
        TEST_FILE = "/mnt/e/Airquality/Datasets/AOD/MCD19A2_061-20260419_230538/MCD19A2.A2025317.h21v09.061.2025318203114.hdf"
        
        # 3. Verify file existence and run tests
        if os.path.exists(TEST_FILE):
            diagnostic_test(TEST_FILE)
        else:
            print(f"❌ File not found at: {TEST_FILE}")
            print("Please check folder name case-sensitivity (e.g., 'Airquality' vs 'airquality').")
    else:
        print("Exiting: Mount required to access test data.")
