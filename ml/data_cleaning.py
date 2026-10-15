"""
Data Quality and Cleaning Module for NYC Taxi Big Data Web Application.

Provides data quality inspection and Dask-based logical filtering functions.
"""

import dask.dataframe as dd
import numpy as np
import pandas as pd


def check_data_quality(df):
    """
    Analyzes missing values, invalid entries, and temporal anomalies across Dask partitions.
    """
    total_rows = len(df)

    # Compute missing values lazily
    null_series = df.isnull().sum().compute()
    missing_table = []
    for col, missing_cnt in null_series.items():
        missing_pct = round((missing_cnt / total_rows) * 100.0, 2)
        missing_table.append({
            "column": col,
            "missing_count": int(missing_cnt),
            "missing_pct": missing_pct
        })

    # Anomaly conditions lazily
    invalid_passenger = (df['passenger_count'] <= 0).sum()
    zero_negative_distance = (df['trip_distance'] <= 0).sum()
    negative_fare = (df['fare_amount'] < 0).sum()
    negative_total = (df['total_amount'] < 0).sum()
    dropoff_before_pickup = (df['tpep_dropoff_datetime'] < df['tpep_pickup_datetime']).sum()
    missing_pickup = df['tpep_pickup_datetime'].isnull().sum()
    missing_dropoff = df['tpep_dropoff_datetime'].isnull().sum()

    # Compute scalar anomaly metrics together
    anomaly_counts = dd.compute(
        invalid_passenger,
        zero_negative_distance,
        negative_fare,
        negative_total,
        dropoff_before_pickup,
        missing_pickup,
        missing_dropoff
    )

    return {
        "total_rows": total_rows,
        "missing_table": missing_table,
        "invalid_passenger_count": int(anomaly_counts[0]),
        "zero_negative_distance": int(anomaly_counts[1]),
        "negative_fare_amount": int(anomaly_counts[2]),
        "negative_total_amount": int(anomaly_counts[3]),
        "dropoff_before_pickup": int(anomaly_counts[4]),
        "missing_pickup_time": int(anomaly_counts[5]),
        "missing_dropoff_time": int(anomaly_counts[6])
    }


def clean_taxi_data(df):
    """
    Applies logic filters to create a cleaned Dask DataFrame.
    Calculates trip_duration_minutes without extraneous feature engineering.
    """
    total_rows = len(df)

    # Filtering rules
    df_clean = df[
        (df['passenger_count'] > 0) &
        (df['trip_distance'] > 0) &
        (df['fare_amount'] >= 0) &
        (df['total_amount'] >= 0) &
        (df['tpep_pickup_datetime'].notnull()) &
        (df['tpep_dropoff_datetime'].notnull()) &
        (df['tpep_dropoff_datetime'] >= df['tpep_pickup_datetime'])
    ]

    # Calculate trip_duration_minutes
    duration_seconds = (df_clean['tpep_dropoff_datetime'] - df_clean['tpep_pickup_datetime']).dt.total_seconds()
    df_clean = df_clean.assign(trip_duration_minutes=duration_seconds / 60.0)

    # Filter valid duration (> 0 and <= 1440 minutes / 24 hours)
    df_clean = df_clean[(df_clean['trip_duration_minutes'] > 0) & (df_clean['trip_duration_minutes'] <= 1440)]

    cleaned_rows = len(df_clean)
    removed_rows = total_rows - cleaned_rows

    cleaning_stats = {
        "original_records": total_rows,
        "invalid_records": removed_rows,
        "cleaned_records": cleaned_rows,
        "removed_records": removed_rows,
        "retention_pct": round((cleaned_rows / total_rows) * 100.0, 2)
    }

    return df_clean, cleaning_stats
