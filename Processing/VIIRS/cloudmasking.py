import netCDF4 as nc

file_path = "/mnt/e/Airquality/Datasets/AOD/RAW_AOD/CLDMSK_L2_VIIRS_SNPP_1-20260506_184021/CLDMSK_L2_VIIRS_SNPP.A2025001.1054.001.2025001231402.nc"

ds = nc.Dataset(file_path)

print("The hidden folders (groups) in this file are:")
for group_name in ds.groups.keys():
    print(f" - {group_name}")
    print("   Variables:", list(ds.groups[group_name].variables.keys()))