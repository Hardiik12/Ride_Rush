"""
Data Manager Accessor Module for Flask Web Application.

Provides cached pipeline data access and rapid fare prediction utilities for Flask route handlers.
"""

from taxi_pipeline import TaxiDataPipeline

_pipeline_instance = None


def get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = TaxiDataPipeline()
    return _pipeline_instance


def get_data(force=False):
    """
    Returns cached or newly computed pipeline metrics dictionary.
    """
    pipeline = get_pipeline()
    return pipeline.run_pipeline(force=force)


def predict_fare(distance):
    """
    Predicts fare amount ($) for a given trip distance in miles
    using the trained Simple Linear Regression model.
    """
    try:
        dist_val = float(distance)
    except (ValueError, TypeError):
        dist_val = 1.0

    pipeline_data = get_data()
    reg = pipeline_data.get("regression", {})
    params = reg.get("model_parameters", {})
    w = float(params.get("weight", 3.25))
    b = float(params.get("bias", 5.50))

    # Calculate predicted fare: y_hat = w * x + b (enforce positive fare >= 2.50 base rate)
    predicted = max(2.50, round(w * dist_val + b, 2))

    return {
        "distance": dist_val,
        "predicted_fare": predicted,
        "weight": w,
        "bias": b,
        "formula": params.get("formula", f"Fare = ${w:.2f} × Distance + ${b:.2f}")
    }


def predict_workbench_fare(distance, model_key="baseline_gd"):
    """
    Predicts fare amount using any selected M2 workbench model
    (baseline_gd, polynomial, ridge, lasso, elastic_net).
    """
    from ml.regression_workbench import predict_workbench_fare_by_model
    from ml.m6_engineering import record_prediction_telemetry
    import time
    t0 = time.time()
    res = predict_workbench_fare_by_model(distance, model_key)
    latency_ms = (time.time() - t0) * 1000.0
    record_prediction_telemetry(latency_ms, model_key)
    return res


def execute_batch_inference(csv_text_or_df, model_key="polynomial"):
    """
    Executes batch fare inference across a tabular batch (CSV or DataFrame).
    """
    from ml.m1_lifecycle import execute_batch_prediction
    return execute_batch_prediction(csv_text_or_df, model_key)


def get_monitoring_dashboard_data():
    """
    Retrieves production telemetry and drift monitoring metrics.
    """
    from ml.m6_engineering import get_monitoring_telemetry
    return get_monitoring_telemetry()


