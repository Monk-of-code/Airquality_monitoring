import pandas as pd
import numpy as np

# 🆕 NEW: weighting function
def compute_weight(hours):
    if hours >= 18:
        return 1.0
    elif hours >= 16:
        return 0.9
    elif hours >= 14:
        return 0.75
    elif hours >= 12:
        return 0.6
    else:
        return 0.0

def generate_daily_metrics(df):
    """Daily averages per device per day (correct grain)."""

    valid_data = df[df['is_valid_clean'] == True].copy()

    #  CHANGED: ONLY group by device + date
    daily = valid_data.groupby(
        ['device_name', 'calc_date']
    ).agg(
        daily_avg_pm25=('cal_pm25', 'mean'),

        #  CHANGED: count UNIQUE hours
        hours_count=('datetime', lambda x: x.dt.floor('h').nunique())
    ).reset_index()
    daily = daily[daily['hours_count'] >= 12].copy()
    #  attach metadata AFTER aggregation
    meta = df[['device_name', 'latitude', 'longitude', 'site_id', 'site_name']].drop_duplicates('device_name')
    daily = daily.merge(meta, on='device_name', how='left')

    #  add weights
    daily["weight"] = daily["hours_count"].apply(compute_weight)

    return daily


def threshold_sensitivity_report(df_daily):
    thresholds = [12, 13, 14, 15, 16, 17, 18]

    report = df_daily.groupby('device_name').size().reset_index(name='total_days_recorded')

    for t in thresholds:
        passed = df_daily[df_daily['hours_count'] >= t] \
            .groupby('device_name').size() \
            .reset_index(name=f'days_at_{t}hr_plus')

        report = report.merge(passed, on='device_name', how='left').fillna(0)

    return report.astype({col: int for col in report.columns if col != 'device_name'})


def annual_coverage_75_report(df_daily):
    thresholds = [12, 13, 14, 15, 16, 17, 18]

    report = df_daily[['device_name']].drop_duplicates()

    #  dynamic total days
    total_days = df_daily['calc_date'].nunique()

    for t in thresholds:
        valid_days = df_daily[df_daily['hours_count'] >= t] \
            .groupby('device_name').size() \
            .reset_index(name=f'days_{t}hr')

        report = report.merge(valid_days, on='device_name', how='left').fillna(0)

        report[f'pct_{t}hr'] = (report[f'days_{t}hr'] / total_days * 100).round(2)
        report[f'status_{t}hr'] = np.where(report[f'pct_{t}hr'] >= 75.0, 'Pass', 'Fail')

    return report