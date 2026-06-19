import pandas as pd
import numpy as np
from filters import stagnation_check_hourly
from qc_logger import log_qc

def run_pipeline(df):
    """
    Executes the Two-Stage Quality Assurance Protocol for hourly sensor data,
    logging the impact of each filter step sequentially.
    """
    # 0. Preparation: Ensure datetime format and chronological sorting for rolling math
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(["device_name", "datetime"]).copy()
    
    rows_before = len(df)
    
    # --- 1. Stagnation Check ---
    df = stagnation_check_hourly(df, hours_threshold=3)
    flagged_stagnation = int(df["stagnation_flag"].sum())
    log_qc("simultaneous_stagnation_check", rows_before, rows_before - flagged_stagnation, flagged_stagnation)
    
    # --- 2. Spatiotemporal Rolling MAD Filter ---
    window_size = 11  
    df['roll_median'] = df.groupby('device_name')['cal_pm25'].transform(
        lambda x: x.rolling(window_size, center=True, min_periods=5).median())
    
    df['abs_dev'] = np.abs(df['cal_pm25'] - df['roll_median'])
    
    df['roll_mad'] = df.groupby('device_name')['abs_dev'].transform(
        lambda x: x.rolling(window_size, center=True, min_periods=5).median())
    df['roll_mad'] = df['roll_mad'].replace(0, 1.0) # Prevent division by zero
    
    df['local_mod_z'] = 0.6745 * df['abs_dev'] / df['roll_mad']
    
    # Flag spikes: Z > 6.0 AND jumps more than 30 ug/m3
    df['mad_spike'] = (df['local_mod_z'] > 6.0) & (df['cal_pm25'] > df['roll_median'] + 30)
    
    flagged_mad = int(df["mad_spike"].sum())
    log_qc("rolling_mad_spike", rows_before, rows_before - flagged_mad, flagged_mad)

    # --- 3. Severe Logical Contradiction (PM2.5 > PM10) ---
    df['severe_pm_contradiction'] = (df['cal_pm25'] - df['cal_pm10'] > 15) & (df['cal_pm25'] / df['cal_pm10'] > 1.3)
    
    # Smart Adjustment: Mathematically correct minor noise issues directly in a new column
    df['cal_pm10_cleaned'] = np.where((df['cal_pm25'] > df['cal_pm10']) & (~df['severe_pm_contradiction']), 
                                      df['cal_pm25'], df['cal_pm10'])
    
    flagged_ratio = int(df["severe_pm_contradiction"].sum())
    log_qc("severe_pm_ratio_contradiction", rows_before, rows_before - flagged_ratio, flagged_ratio)

    # --- 4. Physical Bounds Enforcement ---
    df['physical_outlier'] = (df['cal_pm25'] > 1000) | (df['cal_pm25'] < 0) | (df['cal_pm10'] > 1500) | (df['cal_pm10'] < 0)
    
    flagged_bounds = int(df["physical_outlier"].sum())
    log_qc("physical_bounds_outlier", rows_before, rows_before - flagged_bounds, flagged_bounds)

    # --- 5. Consolidation ---
    # Combine all specific anomaly flags to determine final validity
    df['is_anomaly'] = df['stagnation_flag'] | df['mad_spike'] | df['severe_pm_contradiction'] | df['physical_outlier']
    
    # Keep rows that didn't trigger ANY anomaly for downstream daily metrics/ML
    df['is_valid_clean'] = ~df['is_anomaly']
    
    # Optional: Log the total combined data loss 
    # (This will be slightly less than the sum of the individual flags due to overlapping anomalies)
    total_flagged = int(df['is_anomaly'].sum())
    log_qc("total_hourly_qc_dropped", rows_before, rows_before - total_flagged, total_flagged)

    # Clean up temporary statistical columns to keep memory usage low
    df = df.drop(columns=['roll_median', 'abs_dev', 'roll_mad', 'local_mod_z', 'is_anomaly'])
    
    return df