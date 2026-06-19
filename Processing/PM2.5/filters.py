import pandas as pd
import numpy as np

def stagnation_check_hourly(df, hours_threshold=3, tolerance=0.5, max_gap_allowed=1.5):
    """
    Flags stagnation for hourly PM2.5 and PM10 sensors.
    
    Logic:
    - Sorts by device and time.
    - Resets the timer if values change beyond 'tolerance'.
    - Resets the timer if there is a data gap > 'max_gap_allowed' (e.g., 1.5 hrs).
    - Flags as stagnant only if the stable period lasts >= 'hours_threshold'.
    """
    # Ensure datetime format and sort
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(["device_name", "datetime"]).copy()

    # 1. Calculate the change in PM values between consecutive rows
    diff_pm25 = df.groupby("device_name")["cal_pm25"].diff().abs()
    diff_pm10 = df.groupby("device_name")["cal_pm10"].diff().abs()
    
    # 2. Calculate the time gap between readings (in hours)
    time_gap = df.groupby("device_name")["datetime"].diff().dt.total_seconds() / 3600

    # 3. Identify a "Break" in the stable run
    is_break = (
        (diff_pm25 > tolerance) | 
        (diff_pm10 > tolerance) | 
        (time_gap > max_gap_allowed) | 
        (df.groupby("device_name").cumcount() == 0)
    )
    
    # 4. Create a unique ID for each continuous stable run
    df["run_id"] = is_break.groupby(df["device_name"]).cumsum()

    # 5. Calculate how many hours have passed since the start of the current run
    run_start_time = df.groupby(["device_name", "run_id"])["datetime"].transform("min")
    df["hours_stagnant"] = (df["datetime"] - run_start_time).dt.total_seconds() / 3600

    # 6. Apply the flag: True if the stable duration meets your threshold
    df["stagnation_flag"] = df["hours_stagnant"] >= hours_threshold

    # Cleanup: remove helper columns
    df = df.drop(columns=["run_id", "hours_stagnant"])

    return df


def clean_sensor_data_pipeline(df):
    """
    Executes the complete Two-Stage Quality Assurance Protocol:
    1. Stagnation Check
    2. Rolling MAD Filter
    3. Severe PM Ratio Inconsistency Check
    4. Physical Bounds Enforcement
    """
    print(f"Initial raw rows: {len(df)}")
    
    # --- 1. Stagnation Check ---
    df = stagnation_check_hourly(df, hours_threshold=3, tolerance=0.5, max_gap_allowed=1.5)
    
    # --- 2. Spatiotemporal Rolling MAD Filter ---
    window_size = 11  
    
    # Calculate the localized baseline (Rolling Median)
    df['roll_median'] = df.groupby('device_name')['cal_pm25'].transform(
        lambda x: x.rolling(window_size, center=True, min_periods=5).median())
    
    # Calculate absolute deviation from the local baseline
    df['abs_dev'] = np.abs(df['cal_pm25'] - df['roll_median'])
    
    # Calculate localized variance (Rolling MAD)
    df['roll_mad'] = df.groupby('device_name')['abs_dev'].transform(
        lambda x: x.rolling(window_size, center=True, min_periods=5).median())
    
    # Floor MAD to 1.0 to avoid division by zero during flatlines
    df['roll_mad'] = df['roll_mad'].replace(0, 1.0)
    
    # Compute localized Modified Z-score: 0.6745 * (Value - Median) / MAD
    df['local_mod_z'] = 0.6745 * df['abs_dev'] / df['roll_mad']
    
    # Streamlined MAD Spike Criteria: Z > 6.0 AND jumps more than 30 ug/m3 from local baseline
    df['mad_spike'] = (df['local_mod_z'] > 6.0) & (df['cal_pm25'] > df['roll_median'] + 30)
                      
    # --- 3. Severe Logical Contradiction Filter (PM2.5 > PM10) ---
    # We only flag physical impossibilities where PM2.5 is vastly larger than PM10
    df['severe_pm_contradiction'] = (df['cal_pm25'] - df['cal_pm10'] > 15) & (df['cal_pm25'] / df['cal_pm10'] > 1.3)
    
    # Smart Adjustment: For minor, expected low-cost sensor noise, correct the PM10 up to the PM2.5 value
    df['cal_pm10_cleaned'] = np.where((df['cal_pm25'] > df['cal_pm10']) & (~df['severe_pm_contradiction']), 
                                      df['cal_pm25'], df['cal_pm10'])
    
    # --- 4. Physical Bounds Checks ---
    # Flag absolutely impossible atmospheric values (e.g., negative concentrations or beyond sensor saturation limits)
    df['physical_outlier'] = (df['cal_pm25'] > 1000) | (df['cal_pm25'] < 0) | (df['cal_pm10'] > 1500) | (df['cal_pm10'] < 0)
    
    # --- CONSOLIDATION & CLEANUP ---
    # Combine all anomaly flags into a single deletion trigger
    df['is_anomaly'] = df['stagnation_flag'] | df['mad_spike'] | df['severe_pm_contradiction'] | df['physical_outlier']
    
    # Print Diagnostics before dropping
    print("\n--- FILTER DIAGNOSTICS ---")
    print(f"Stagnant (Stuck Sensor) Rows: {df['stagnation_flag'].sum()}")
    print(f"MAD Spikes (>30 jump & Z>6.0): {df['mad_spike'].sum()}")
    print(f"Severe PM Ratio Contradictions: {df['severe_pm_contradiction'].sum()}")
    print(f"Physical Bound Outliers (<0 or >1000): {df['physical_outlier'].sum()}")
    print(f"Total Anomalies Dropped: {df['is_anomaly'].sum()}")
    
    # Filter out anomalies and drop the temporary calculation columns
    df_cleaned = df[~df['is_anomaly']].drop(columns=[
        'roll_median', 'abs_dev', 'roll_mad', 'local_mod_z', 
        'mad_spike', 'stagnation_flag', 'severe_pm_contradiction', 'physical_outlier', 'is_anomaly'
    ])
    
    print(f"Final clean rows retained: {len(df_cleaned)}")
    
    return df_cleaned

# ==========================================
# HOW TO RUN IT IN YOUR COLAB NOTEBOOK
# ==========================================
# df = pd.read_csv('/content/your_hourly_data.csv')
# df_clean = clean_sensor_data_pipeline(df)
# df_clean.to_csv('/content/your_hourly_data_cleaned.csv', index=False)