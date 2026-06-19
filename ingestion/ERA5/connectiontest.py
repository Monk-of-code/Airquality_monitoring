import cdsapi
c = cdsapi.Client()
try:
    print("Testing connection...")
    # This just checks if you can connect to the server
    c.retrieve("reanalysis-era5-land", {
        "variable": "2m_temperature",
        "year": "2024",
        "month": "01",
        "day": "01",
        "time": "12:00",
        "area": [-1.2, 36.7, -1.3, 36.8], # Tiny bit of Nairobi
        "data_format": "netcdf",
    }).download("test.nc")
    print("Success! Your .cdsapirc is working.")
except Exception as e:
    print(f"Error: {e}")
