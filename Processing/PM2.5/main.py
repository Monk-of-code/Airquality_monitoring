import pandas as pd
from db import read_sql, write_table
from config import db_config
from qc_logger import init_qc_log
from pipeline import run_pipeline
from reports import generate_daily_metrics, threshold_sensitivity_report, annual_coverage_75_report

def load_from_bronze():
    query = f"""
        SELECT datetime, device_name, site_id, site_name, latitude, longitude,
               pm2_5_calibrated_value AS cal_pm25, pm10_calibrated_value AS cal_pm10
        FROM {db_config.schema}.{db_config.bronze_cal_table}
    """
    return read_sql(query)

def main():
    init_qc_log()
    df = load_from_bronze()

    #  enforce UTC
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    #  convert to Nairobi time
    df["datetime_eat"] = df["datetime"].dt.tz_convert("Africa/Nairobi")

    #  correct local date
    df["calc_date"] = df["datetime_eat"].dt.date

    # enforce 1 row per device-hour
    df = (
        df.groupby(["device_name", "datetime"], as_index=False)
        .agg({
            "cal_pm25": "mean",
            "cal_pm10": "mean",
            "latitude": "first",
            "longitude": "first",
            "site_id": "first",
            "site_name": "first",
            "calc_date": "first"
        })
    )

    df_processed = run_pipeline(df)
    
    df_daily = generate_daily_metrics(df_processed)
    threshold_rpt = threshold_sensitivity_report(df_daily)
    annual_rpt = annual_coverage_75_report(df_daily)
    
    write_table(df_daily, db_config.daily_avg_table)
    write_table(threshold_rpt, db_config.threshold_rpt_table)
    write_table(annual_rpt, db_config.annual_rpt_table)
    print("QC Pipeline Finished.")

if __name__ == "__main__":
    main()