"""
Data Loader Module for NYC Taxi Big Data Web Application.

Provides Dask-based lazy dataset ingestion and schema inspection functions
for Parquet datasets.
"""

import os
import requests
import dask.dataframe as dd
import pandas as pd

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "yellow_tripdata_2025-01.parquet")
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"


def ensure_dataset_exists(path=DEFAULT_DATA_PATH, url=DATA_URL, auto_download=True):
    """
    Validates dataset existence; downloads automatically if missing.
    """
    if os.path.exists(path):
        return True

    if auto_download:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            print(f"[INFO] Downloading dataset from {url}...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print(f"[SUCCESS] Dataset saved to {path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed downloading dataset: {e}")
            if os.path.exists(path):
                os.remove(path)
            return False
    return False


def load_taxi_data(path=DEFAULT_DATA_PATH, auto_download=True):
    """
    Loads NYC Yellow Taxi Parquet dataset lazily into a Dask DataFrame.
    """
    if not os.path.exists(path):
        if not ensure_dataset_exists(path=path, auto_download=auto_download):
            raise FileNotFoundError(f"Parquet dataset not found at '{path}'. Place 'yellow_tripdata_2025-01.parquet' in data/.")

    return dd.read_parquet(path)


def get_dataset_info(df, path=DEFAULT_DATA_PATH):
    """
    Extracts structural metadata and descriptive summary statistics using Dask.
    """
    total_rows = len(df)
    num_cols = len(df.columns)
    num_partitions = df.npartitions

    cols_info = []
    for col in df.columns:
        cols_info.append({
            "name": col,
            "dtype": str(df[col].dtype)
        })

    # Dask lazy compute for head(10) with string conversion for Datetimes
    sample_df = df.head(10).copy()
    for col in sample_df.columns:
        if pd.api.types.is_datetime64_any_dtype(sample_df[col]) or isinstance(sample_df[col].dtype, pd.DatetimeTZDtype):
            sample_df[col] = sample_df[col].astype(str)

    # Compute summary statistics lazily for numerical columns
    numerical_cols = ['passenger_count', 'trip_distance', 'fare_amount', 'tip_amount', 'total_amount']
    existing_num_cols = [c for c in numerical_cols if c in df.columns]
    
    stats_df = df[existing_num_cols].describe().compute().round(2)
    stats_dict = stats_df.to_dict()

    return {
        "dataset_name": os.path.basename(path),
        "file_format": "Parquet",
        "file_path": path,
        "total_rows": total_rows,
        "num_columns": num_cols,
        "num_partitions": num_partitions,
        "shape": (total_rows, num_cols),
        "columns": cols_info,
        "sample_rows": sample_df.to_dict(orient="records"),
        "statistics": stats_dict
    }
