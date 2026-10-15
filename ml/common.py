"""
RideRush Common Machine Learning Utilities and Shared Infrastructure.

Provides standardized, leakage-free data partitioning, feature extraction,
schema validation, and metric computation across M1-M6 modules.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "reports")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")
CHARTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "charts")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

TYPE_MAP = {
    "float": float,
    "int": int,
    "str": str
}

# Schema definitions for inference validation
FEATURE_SCHEMA = {
    "trip_distance": {"type": "float", "min": 0.05, "max": 100.0, "required": True},
    "passenger_count": {"type": "int", "min": 1, "max": 8, "required": False, "default": 1},
    "pickup_hour": {"type": "int", "min": 0, "max": 23, "required": False, "default": 12},
    "pickup_dayofweek": {"type": "int", "min": 0, "max": 6, "required": False, "default": 2},
    "trip_duration_min": {"type": "float", "min": 0.5, "max": 180.0, "required": False, "default": 12.0}
}


def extract_representative_ml_sample(df_clean, sample_frac=0.03, random_state=42):
    """
    Extracts a bounded, reproducible Pandas sample from Dask df_clean
    with engineered features for multi-model ML modules.
    
    Returns:
        pd.DataFrame: Cleaned, feature-engineered tabular dataset.
    """
    # Sample fraction through Dask streaming without materializing the 2.8M rows
    needed_cols = [
        'trip_distance', 'fare_amount', 'tip_amount', 'total_amount',
        'passenger_count', 'PULocationID', 'DOLocationID',
        'tpep_pickup_datetime', 'tpep_dropoff_datetime'
    ]
    avail_cols = [c for c in needed_cols if c in df_clean.columns]
    
    sample_dask = df_clean[avail_cols].sample(frac=sample_frac, random_state=random_state)
    pdf = sample_dask.compute().copy()
    
    # Feature Engineering
    if 'tpep_pickup_datetime' in pdf.columns and 'tpep_dropoff_datetime' in pdf.columns:
        pdf['tpep_pickup_datetime'] = pd.to_datetime(pdf['tpep_pickup_datetime'])
        pdf['tpep_dropoff_datetime'] = pd.to_datetime(pdf['tpep_dropoff_datetime'])
        duration = (pdf['tpep_dropoff_datetime'] - pdf['tpep_pickup_datetime']).dt.total_seconds() / 60.0
        pdf['trip_duration_min'] = duration
        pdf['pickup_hour'] = pdf['tpep_pickup_datetime'].dt.hour
        pdf['pickup_dayofweek'] = pdf['tpep_pickup_datetime'].dt.dayofweek
    else:
        pdf['trip_duration_min'] = pdf['trip_distance'] * 3.5 + 2.0
        pdf['pickup_hour'] = 12
        pdf['pickup_dayofweek'] = 2

    # Clean domain filters
    mask = (
        (pdf['trip_distance'] >= 0.1) & (pdf['trip_distance'] <= 40.0) &
        (pdf['fare_amount'] >= 2.50) & (pdf['fare_amount'] <= 150.0) &
        (pdf['trip_duration_min'] >= 0.5) & (pdf['trip_duration_min'] <= 120.0)
    )
    if 'passenger_count' in pdf.columns:
        mask = mask & (pdf['passenger_count'] >= 1) & (pdf['passenger_count'] <= 6)
    
    clean_sample = pdf[mask].copy()

    # Targets:
    # 1. Regression target: fare_amount
    # 2. Binary classification target: is_high_tip (tip_fraction > 0.18)
    clean_sample['tip_fraction'] = np.where(
        clean_sample['fare_amount'] > 0,
        clean_sample['tip_amount'] / clean_sample['fare_amount'],
        0.0
    )
    clean_sample['is_high_tip'] = (clean_sample['tip_fraction'] >= 0.18).astype(int)
    
    # 3. Multiclass classification target: fare_tier (0: Low < $12, 1: Medium $12-$25, 2: High > $25)
    clean_sample['fare_tier'] = pd.cut(
        clean_sample['fare_amount'],
        bins=[-np.inf, 12.0, 25.0, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    return clean_sample


def prepare_train_val_test_splits(df_sample, feature_cols, target_col, test_size=0.15, val_size=0.15, random_state=42):
    """
    Creates strict 3-way partition: 70% Train, 15% Validation, 15% Test.
    Ensures zero leakage across sets.
    """
    X = df_sample[feature_cols].copy()
    y = df_sample[target_col].copy()

    # First split off test set (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Next split train and val from remainder (val is 15% of total -> 0.15 / 0.85 approx 0.1765)
    val_ratio_of_rem = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_of_rem, random_state=random_state
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "counts": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
            "total": len(X)
        }
    }


def validate_inference_payload(payload):
    """
    Validates single inference input against FEATURE_SCHEMA.
    Returns cleaned dict with types coerced, or raises ValueError.
    """
    cleaned = {}
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON dictionary.")

    for feat, rules in FEATURE_SCHEMA.items():
        type_fn = TYPE_MAP.get(rules.get("type", "float"), float)
        if feat in payload:
            val = payload[feat]
            try:
                coerced = type_fn(val)
            except (ValueError, TypeError):
                raise ValueError(f"Feature '{feat}' must be convertible to {rules.get('type', 'float')}.")
            
            if "min" in rules and coerced < rules["min"]:
                raise ValueError(f"Feature '{feat}' ({coerced}) below minimum allowed ({rules['min']}).")
            if "max" in rules and coerced > rules["max"]:
                raise ValueError(f"Feature '{feat}' ({coerced}) exceeds maximum allowed ({rules['max']}).")
            cleaned[feat] = coerced
        else:
            if rules.get("required", False):
                raise ValueError(f"Missing required feature: '{feat}'")
            cleaned[feat] = rules.get("default", 0.0)

    return cleaned
