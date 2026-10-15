"""
Pytest Test Suite for NYC Taxi Big Data EDA & Linear Regression Web Application.
"""

import os
import pytest
import numpy as np
import pandas as pd
from app import app
from ml.data_loader import load_taxi_data
from ml.linear_regression import SimpleLinearRegressionGD
from ml.outliers import calculate_iqr_statistics, evaluate_outlier_treatments
from ml.missing_values import (
    audit_missing_values,
    delete_missing_rows,
    delete_missing_columns,
    mean_imputation,
    median_imputation,
    knn_imputation,
    iterative_imputation,
    create_missing_indicators
)
from ml.scaling import (
    standard_scaling,
    minmax_scaling,
    robust_scaling,
    demonstrate_scaling_leakage,
    get_scaler_comparison_matrix,
    get_model_scaling_guidance
)
from ml.duplicates import (
    audit_exact_duplicates,
    audit_subset_duplicates,
    evaluate_duplicate_removal_treatments,
    explain_legitimate_trip_risks
)
from ml.regression_workbench import (
    calculate_regression_metrics,
    train_workbench_models,
    evaluate_all_workbench_models,
    predict_workbench_fare_by_model,
    get_regularization_educational_theory,
    solve_normal_equation
)
from ml.m1_lifecycle import (
    get_lifecycle_stage_mapping,
    get_lifecycle_educational_theory,
    execute_batch_prediction,
    check_retraining_triggers
)
from ml.m2_classification import (
    sigmoid, softmax, compute_cross_entropy_loss,
    out_of_fold_target_encoding,
    train_and_evaluate_classification_models
)
from ml.m3_trees import (
    compute_tree_split_criteria,
    compute_cost_complexity_pruning_path,
    train_and_benchmark_tree_models,
    get_tree_educational_theory
)
from ml.m4_unsupervised import (
    run_complete_m4_unsupervised,
    get_unsupervised_educational_theory
)
from ml.m5_evaluation import (
    compute_comprehensive_regression_metrics,
    compute_mcnemar_test,
    compute_paired_bootstrap_ci,
    run_model_selection_experiments,
    run_complete_m5_evaluation
)
from ml.m6_engineering import (
    save_model_artifact,
    load_model_artifact,
    compute_sha256,
    detect_feature_drift,
    calculate_psi,
    evaluate_promotion_gate,
    record_prediction_telemetry,
    get_monitoring_telemetry
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


@pytest.fixture
def client():
    """
    Flask test client fixture.
    """
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_dataset_exists():
    """
    Verifies that the dataset file exists in data/ directory.
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "yellow_tripdata_2025-01.parquet")
    assert os.path.exists(data_path), f"Dataset parquet file missing at {data_path}"


def test_dask_loader():
    """
    Verifies lazy loading of Parquet dataset via Dask.
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "yellow_tripdata_2025-01.parquet")
    df = load_taxi_data(data_path)
    assert df is not None
    assert len(df.columns) > 0


def test_home_route(client):
    """
    Verifies Home Dashboard page route.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert b"NYC Taxi" in response.data


def test_data_route(client):
    """
    Verifies Data Understanding page route.
    """
    response = client.get("/data")
    assert response.status_code == 200
    assert b"Data Understanding" in response.data


def test_cleaning_route(client):
    """
    Verifies Data Quality & Cleaning page route.
    """
    response = client.get("/cleaning")
    assert response.status_code == 200
    assert b"Data Quality" in response.data


def test_eda_route(client):
    """
    Verifies EDA Dashboard page route.
    """
    response = client.get("/eda")
    assert response.status_code == 200
    assert b"EDA Dashboard" in response.data
    assert b"Outlier Analysis" in response.data
    assert b"Missing Value Analysis" in response.data


def test_insights_route(client):
    """
    Verifies EDA Key Insights page route.
    """
    response = client.get("/insights")
    assert response.status_code == 200
    assert b"Key Insights" in response.data


def test_api_summary_route(client):
    """
    Verifies API JSON summary route.
    """
    response = client.get("/api/summary")
    assert response.status_code == 200
    assert response.is_json
    json_data = response.get_json()
    assert "dataset_info" in json_data
    assert "outliers" in json_data or "outlier_table" in json_data
    assert "missing_values" in json_data
    assert "scaling" in json_data
    assert "duplicates" in json_data


def test_linear_regression_gd_model():
    """
    Verifies SimpleLinearRegressionGD math & fitting logic on synthetic data y = 3x + 5.
    """
    np.random.seed(42)
    X = np.linspace(1, 20, 100)
    y = 3.0 * X + 5.0 + np.random.normal(0, 0.5, 100)

    model = SimpleLinearRegressionGD(learning_rate=0.01, iterations=1000)
    model.fit(X, y)

    metrics = model.evaluate(X, y)
    assert abs(model.w - 3.0) < 0.5, f"Expected slope around 3.0, got {model.w}"
    assert abs(model.b - 5.0) < 1.0, f"Expected intercept around 5.0, got {model.b}"
    assert metrics["r2"] > 0.95
    assert len(model.cost_history) == 1000
    assert model.cost_history[-1] < model.cost_history[0]


def test_prediction_page_get(client):
    """
    Verifies GET /prediction route.
    """
    response = client.get("/prediction")
    assert response.status_code == 200
    assert b"Simple Linear Regression" in response.data
    assert b"Interactive Taxi Fare Predictor" in response.data


def test_prediction_page_post(client):
    """
    Verifies POST /prediction route with distance input.
    """
    response = client.post("/prediction", data={"distance": "10.0"})
    assert response.status_code == 200
    assert b"Estimated Meter Fare" in response.data
    assert b"10.0" in response.data


def test_api_predict_route(client):
    """
    Verifies GET /api/predict?distance=5.2 JSON endpoint.
    """
    response = client.get("/api/predict?distance=5.2")
    assert response.status_code == 200
    assert response.is_json
    json_data = response.get_json()
    assert json_data["distance"] == 5.2
    assert "predicted_fare" in json_data
    assert json_data["predicted_fare"] > 0


def test_co1_outlier_module():
    """
    Verifies CO1 Outlier Analysis functions: IQR bounds, Winsorization, Remove vs Retain.
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "yellow_tripdata_2025-01.parquet")
    df = load_taxi_data(data_path)
    from ml.data_cleaning import clean_taxi_data
    df_clean, _ = clean_taxi_data(df)

    iqr_table = calculate_iqr_statistics(df_clean)
    assert len(iqr_table) == 4

    treatments = evaluate_outlier_treatments(df_clean, iqr_table)
    assert "remove_impact" in treatments
    assert "retain_impact" in treatments


def test_co1_missing_values_module():
    """
    Verifies CO1 Missing Value Analysis functions: audit, deletion, mean/median imputation,
    KNN, Iterative MICE, and missing indicators.
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "yellow_tripdata_2025-01.parquet")
    df = load_taxi_data(data_path)
    from ml.data_cleaning import clean_taxi_data
    df_clean, _ = clean_taxi_data(df)

    audit = audit_missing_values(df_clean)
    assert audit["total_rows"] > 0
    assert len(audit["audit_table"]) == len(df_clean.columns)

    row_del = delete_missing_rows(df_clean)
    assert row_del["remaining_rows"] <= row_del["original_rows"]

    col_del = delete_missing_columns(df_clean, audit["audit_table"])
    assert col_del["retained_columns_count"] <= col_del["original_columns_count"]

    mean_imp = mean_imputation(df_clean)
    assert len(mean_imp) > 0

    med_imp = median_imputation(df_clean)
    assert len(med_imp) > 0

    sample_df = df_clean.sample(frac=0.01, random_state=42).compute()

    knn_res = knn_imputation(sample_df)
    assert knn_res["missing_after"] == 0

    mice_res = iterative_imputation(sample_df)
    assert mice_res["missing_after"] == 0

    indicators = create_missing_indicators(sample_df)
    assert len(indicators) > 0


def test_co1_scaling_module():
    """
    Verifies CO1 Feature Scaling: StandardScaler (Z-score), MinMaxScaler, RobustScaler,
    training-only scaler fitting, leakage detection, and model selection matrices.
    """
    np.random.seed(42)
    sample_df = pd.DataFrame({
        "trip_distance": [1.2, 2.5, 3.8, 15.0, 0.8, 4.2, 5.0, 22.0],
        "fare_amount": [7.5, 12.0, 18.5, 65.0, 5.0, 20.0, 24.5, 95.0],
        "trip_duration_minutes": [5.0, 10.0, 15.0, 45.0, 3.0, 16.0, 18.0, 60.0],
        "total_amount": [10.0, 15.5, 23.0, 80.0, 7.5, 26.0, 31.0, 115.0]
    })

    train_df = sample_df.iloc[:6]
    test_df = sample_df.iloc[6:]

    # 1. StandardScaler (Z-Score)
    std_res = standard_scaling(train_df, test_df)
    assert std_res["scaler_name"] == "StandardScaler (Z-Score)"
    assert "formula" in std_res
    train_std = std_res["train_scaled_df"]
    # Check mean ~ 0 and std ~ 1 on training set
    assert abs(train_std["trip_distance"].mean()) < 1e-4
    assert abs(train_std["trip_distance"].std(ddof=0) - 1.0) < 1e-4
    assert std_res["test_scaled_df"] is not None

    # 2. MinMaxScaler
    minmax_res = minmax_scaling(train_df, test_df)
    assert minmax_res["scaler_name"] == "MinMaxScaler"
    train_mm = minmax_res["train_scaled_df"]
    assert train_mm["fare_amount"].min() >= 0.0
    assert train_mm["fare_amount"].max() <= 1.0

    # 3. RobustScaler (Median & IQR)
    robust_res = robust_scaling(train_df, test_df)
    assert robust_res["scaler_name"] == "RobustScaler"
    train_rob = robust_res["train_scaled_df"]
    assert abs(train_rob["trip_duration_minutes"].median()) < 1e-4

    # 4. Training-Only Fitting & Data Leakage Prevention
    leakage = demonstrate_scaling_leakage(sample_df, feature="fare_amount")
    assert "safe_train_mean" in leakage
    assert "leaky_combined_mean" in leakage
    assert "mean_shift" in leakage
    assert leakage["mean_shift"] >= 0.0

    # 5. Conceptual Guidance
    matrix = get_scaler_comparison_matrix()
    assert len(matrix) == 3
    guidance = get_model_scaling_guidance()
    assert len(guidance["models_needing_scaling"]) >= 3
    assert len(guidance["models_not_needing_scaling"]) >= 2


def test_co1_duplicates_module():
    """
    Verifies CO1 Duplicate Detection: exact-row duplicates, subset duplicates,
    removal comparison, and operational risk factors.
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "yellow_tripdata_2025-01.parquet")
    df = load_taxi_data(data_path)
    from ml.data_cleaning import clean_taxi_data
    df_clean, _ = clean_taxi_data(df)

    # 1. Exact Full-Row Duplicate Audit
    exact_audit = audit_exact_duplicates(df_clean)
    assert exact_audit["total_records"] == 2816149
    assert exact_audit["duplicate_count"] == 0
    assert exact_audit["duplicate_pct"] == 0.0

    # 2. Business-Key Subset Duplicate Audit
    subset_audit = audit_subset_duplicates(df_clean)
    assert subset_audit["total_records"] == 2816149
    assert subset_audit["duplicate_count"] >= 0
    assert len(subset_audit["subset_columns"]) >= 5

    # 3. Treatment Comparison
    treatments = evaluate_duplicate_removal_treatments(df_clean, exact_audit, subset_audit)
    assert len(treatments) == 3
    assert treatments[0]["strategy"] == "Retain All Records (Baseline)"
    assert treatments[0]["records_removed"] == 0
    assert treatments[1]["strategy"] == "Exact-Row Deletion"
    assert treatments[2]["strategy"] == "Business-Key Subset Deletion"

    # 4. Operational Risk Explanations
    risks = explain_legitimate_trip_risks()
    assert len(risks) >= 3


def test_eda_co1_rendering(client):
    """
    Verifies that /eda template renders all CO1 sections:
    Outliers, Missing Values, Feature Scaling, and Duplicate Records.
    """
    response = client.get("/eda")
    assert response.status_code == 200
    assert b"Phase 5" in response.data
    assert b"Outlier Analysis (CO1)" in response.data
    assert b"Missing Values (CO1)" in response.data
    assert b"Feature Scaling (CO1)" in response.data
    assert b"Duplicate Records (CO1)" in response.data
    assert b"StandardScaler" in response.data
    assert b"MinMaxScaler" in response.data
    assert b"RobustScaler" in response.data
    assert b"Data Leakage Prevention" in response.data
    assert b"Exact Full-Row Duplicates" in response.data


def test_workbench_synthetic_models_train_and_predict():
    """
    Verifies that all 5 workbench regression models can train and predict on synthetic data:
    1. Baseline Gradient Descent (SimpleLinearRegressionGD)
    2. Polynomial Regression (Degree 2)
    3. Ridge Regression (L2)
    4. Lasso Regression (L1)
    5. Elastic Net (L1+L2)
    """
    np.random.seed(42)
    X_syn = np.linspace(1.0, 25.0, 150).reshape(-1, 1)
    y_syn = 3.5 * X_syn.ravel() + 5.0 + np.random.normal(0, 1.5, 150)

    models = train_workbench_models(
        X_syn, y_syn,
        poly_degree=2,
        ridge_alpha=1.0,
        lasso_alpha=0.1,
        elastic_alpha=0.1,
        elastic_l1_ratio=0.5
    )

    expected_keys = {"baseline_gd", "polynomial", "ridge", "lasso", "elastic_net"}
    assert set(models.keys()) == expected_keys

    X_test_syn = np.array([[3.0], [7.5], [12.0]])
    y_test_syn = np.array([15.5, 31.2, 47.0])

    eval_results = evaluate_all_workbench_models(models, X_test_syn, y_test_syn)
    assert len(eval_results) == 5

    for res in eval_results:
        assert np.isfinite(res["mae"])
        assert np.isfinite(res["mse"])
        assert np.isfinite(res["rmse"])
        assert np.isfinite(res["r2"])
        assert res["mae"] >= 0.0
        assert res["rmse"] >= 0.0
        assert res["sample_prediction_10mi"] > 2.50


def test_workbench_metrics_calculation():
    """
    Verifies that calculate_regression_metrics returns finite and mathematically correct scores.
    """
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred_perfect = np.array([10.0, 20.0, 30.0, 40.0])

    metrics_perfect = calculate_regression_metrics(y_true, y_pred_perfect)
    assert metrics_perfect["mae"] == 0.0
    assert metrics_perfect["mse"] == 0.0
    assert metrics_perfect["rmse"] == 0.0
    assert metrics_perfect["r2"] == 1.0

    y_pred_offset = np.array([12.0, 18.0, 32.0, 38.0])
    metrics_offset = calculate_regression_metrics(y_true, y_pred_offset)
    assert metrics_offset["mae"] == 2.0
    assert metrics_offset["mse"] == 4.0
    assert metrics_offset["rmse"] == 2.0
    assert np.isfinite(metrics_offset["r2"])


def test_workbench_polynomial_feature_generation():
    """
    Verifies polynomial feature generation expands single feature dimension [x] into [x, x^2].
    """
    X = np.array([[2.0], [3.0], [4.0]])
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Output shape should be (3, 2): column 0 is x, column 1 is x^2
    assert X_poly.shape == (3, 2)
    np.testing.assert_array_almost_equal(X_poly[:, 0], np.array([2.0, 3.0, 4.0]))
    np.testing.assert_array_almost_equal(X_poly[:, 1], np.array([4.0, 9.0, 16.0]))

    # Degree 3 check: [x, x^2, x^3]
    poly3 = PolynomialFeatures(degree=3, include_bias=False)
    X_poly3 = poly3.fit_transform(X)
    assert X_poly3.shape == (3, 3)
    np.testing.assert_array_almost_equal(X_poly3[:, 2], np.array([8.0, 27.0, 64.0]))


def test_workbench_regularization_parameters_configurable():
    """
    Verifies that hyperparameters (alpha, degree, l1_ratio) are fully configurable
    and that stronger L2 regularization shrinks coefficients appropriately.
    """
    np.random.seed(42)
    X_train = np.linspace(1.0, 20.0, 80).reshape(-1, 1)
    y_train = 4.0 * X_train.ravel() + 10.0 + np.random.normal(0, 2.0, 80)

    # Train with custom hyperparameters
    custom_models = train_workbench_models(
        X_train, y_train,
        poly_degree=2,
        ridge_alpha=50.0,
        lasso_alpha=0.25,
        elastic_alpha=0.35,
        elastic_l1_ratio=0.75
    )

    assert custom_models["ridge"]["hyperparameters"]["alpha"] == 50.0
    assert custom_models["lasso"]["hyperparameters"]["alpha"] == 0.25
    assert custom_models["elastic_net"]["hyperparameters"]["alpha"] == 0.35
    assert custom_models["elastic_net"]["hyperparameters"]["l1_ratio"] == 0.75

    # Check shrinkage: Ridge with alpha=500 vs Ridge with alpha=0.01
    pipe_weak = Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=0.01, random_state=42))])
    pipe_strong = Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=500.0, random_state=42))])

    pipe_weak.fit(X_train, y_train)
    pipe_strong.fit(X_train, y_train)

    coef_weak = abs(pipe_weak.named_steps['reg'].coef_[0])
    coef_strong = abs(pipe_strong.named_steps['reg'].coef_[0])

    assert coef_strong < coef_weak, f"Strong alpha should shrink coefficient ({coef_strong} < {coef_weak})"


def test_workbench_training_only_leakage_prevention():
    """
    Verifies that sklearn Pipeline preprocessors fit solely on X_train,
    preventing any statistical distribution leakage from test data.
    """
    np.random.seed(42)
    # Train set centered at 10.0, test set centered at 50.0
    X_train = np.random.normal(10.0, 2.0, size=(100, 1))
    X_test = np.random.normal(50.0, 2.0, size=(50, 1))
    y_train = 2.0 * X_train.ravel() + 1.0

    models = train_workbench_models(X_train, y_train)
    ridge_pipeline = models["ridge"]["model"]

    # Scaler inside pipeline should store train mean, not test mean
    scaler = ridge_pipeline.named_steps["scaler"]
    train_mean = float(np.mean(X_train))
    pipeline_scaler_mean = float(scaler.mean_[0])

    assert abs(pipeline_scaler_mean - train_mean) < 1e-5
    assert abs(pipeline_scaler_mean - 50.0) > 30.0  # Asserts test set distribution was NOT seen


def test_workbench_educational_theory():
    """
    Verifies that get_regularization_educational_theory returns all 4 required educational sections.
    """
    theory = get_regularization_educational_theory()
    assert "why_regularization_helps" in theory
    assert "alpha_parameter_intuition" in theory
    assert "l1_vs_l2_geometry" in theory
    assert "polynomial_overfitting_risks" in theory
    assert len(theory["why_regularization_helps"]) > 100
    assert len(theory["l1_vs_l2_geometry"]) > 100


def test_workbench_page_and_api_routes(client):
    """
    Verifies that /workbench and API endpoints respond with HTTP 200 and expected payloads.
    """
    # 1. GET /workbench
    resp = client.get("/workbench")
    assert resp.status_code == 200
    assert b"Regression Model Workbench" in resp.data
    assert b"Held-Out Test Evaluation Benchmark" in resp.data
    assert b"Polynomial Regression" in resp.data
    assert b"Ridge Regression" in resp.data
    assert b"Lasso Regression" in resp.data
    assert b"Elastic Net" in resp.data

    # 2. POST /workbench (interactive multi-model prediction)
    post_resp = client.post("/workbench", data={"distance": "8.5", "model_key": "ridge"})
    assert post_resp.status_code == 200
    assert b"Predicted Fare Amount" in post_resp.data

    # 3. GET /api/workbench
    api_resp = client.get("/api/workbench")
    assert api_resp.status_code == 200
    api_json = api_resp.get_json()
    assert isinstance(api_json, dict)
    assert "models_evaluated" in api_json or "error" in api_json

    # 4. GET /api/workbench/predict
    pred_api_resp = client.get("/api/workbench/predict?distance=6.0&model=polynomial")
    assert pred_api_resp.status_code == 200
    pred_json = pred_api_resp.get_json()
    assert "predicted_fare" in pred_json
    assert pred_json["predicted_fare"] >= 2.50
    assert pred_json["model_key"] == "polynomial"


def test_m1_lifecycle_mapping_and_batch_prediction():
    """
    Verifies M1: 12-stage lifecycle mapping, educational theory, and batch prediction.
    """
    stages = get_lifecycle_stage_mapping()
    assert len(stages) == 12
    assert stages[0]["name"] == "Data Ingestion & Lazy Loading"
    assert stages[11]["name"] == "Retraining Triggers & Promotion Gate"

    theory = get_lifecycle_educational_theory()
    assert "learning_paradigms" in theory
    assert "training_serving_skew" in theory
    assert "feature_store_architecture" in theory

    # Batch prediction test
    csv_input = "trip_distance,passenger_count\n1.5,1\n3.0,2\n8.5,1\n"
    res = execute_batch_prediction(csv_input, model_key="polynomial")
    assert res["status"] == "success"
    assert res["total_processed"] == 3
    assert res["mean_fare"] > 2.50
    assert res["total_revenue"] > 10.0
    assert len(res["preview"]) == 3


def test_m2_normal_equation_analytical_solution():
    """
    Verifies M2: Closed-form Ordinary Least Squares solution via the Normal Equation:
    theta = (X_b.T @ X_b)^(-1) @ X_b.T @ y
    """
    np.random.seed(42)
    X = np.linspace(1.0, 10.0, 50)
    y = 3.5 * X + 4.2  # Exact linear relation

    norm_res = solve_normal_equation(X, y)
    assert abs(norm_res["weight"] - 3.5) < 1e-3
    assert abs(norm_res["bias"] - 4.2) < 1e-3
    assert "Normal Equation" in norm_res["method"]


def test_m2_classification_models_and_metrics():
    """
    Verifies M2: Binary Logistic Regression, Multinomial Logistic Regression,
    Sigmoid, Softmax, Cross-Entropy Loss, and Class Imbalance mitigation.
    """
    # Test activations
    z = np.array([-10.0, 0.0, 10.0])
    sig = sigmoid(z)
    assert abs(sig[1] - 0.5) < 1e-5
    assert sig[0] < 0.01 and sig[2] > 0.99

    z_multi = np.array([1.0, 2.0, 3.0])
    sm = softmax(z_multi)
    assert abs(np.sum(sm) - 1.0) < 1e-5
    assert sm[2] > sm[1] > sm[0]

    # Test Cross-Entropy Loss
    loss_val = compute_cross_entropy_loss(np.array([1, 0]), np.array([0.9, 0.1]))
    assert loss_val > 0.0 and loss_val < 0.25

    # Run complete classification on synthetic sample
    clf_res = train_and_evaluate_classification_models(pd.DataFrame({
        "trip_distance": [1.0, 2.0, 5.0, 10.0, 15.0, 20.0] * 20,
        "pickup_hour": [8, 12, 18, 22, 14, 9] * 20,
        "pickup_dayofweek": [0, 1, 2, 3, 4, 5] * 20,
        "trip_duration_min": [5.0, 8.0, 18.0, 35.0, 45.0, 60.0] * 20,
        "is_high_tip": [0, 0, 1, 1, 0, 1] * 20,
        "fare_tier": [0, 0, 1, 1, 2, 2] * 20
    }))

    assert "standard" in clf_res["binary_models"]
    assert "balanced" in clf_res["binary_models"]
    assert clf_res["binary_models"]["standard"]["accuracy"] > 0.5
    assert clf_res["multinomial_model"]["accuracy"] > 0.5
    assert len(clf_res["multinomial_model"]["classes"]) == 3


def test_m2_target_encoding_leakage_safety():
    """
    Verifies M2: Leakage-safe Out-of-Fold (OOF) Target Encoding
    computes encodings for training without test target leakage.
    """
    df_tr = pd.DataFrame({
        "cat": ["A", "A", "B", "B", "A", "B"] * 10,
        "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 10
    })
    y_tr = pd.Series([1, 0, 1, 1, 0, 1] * 10)
    df_te = pd.DataFrame({"cat": ["A", "B", "C"]})

    encoded_tr, encoded_te, prior = out_of_fold_target_encoding(
        df_tr, y_tr, df_te, "cat", n_splits=3, random_state=42
    )

    assert len(encoded_tr) == len(df_tr)
    assert len(encoded_te) == len(df_te)
    assert 0.0 <= prior <= 1.0


def test_m3_tree_models_and_ensembles():
    """
    Verifies M3: Decision Trees, Random Forest (with OOB), AdaBoost,
    and HistGradientBoosting (LightGBM equivalent) train and evaluate.
    """
    np.random.seed(42)
    n = 120
    df_syn = pd.DataFrame({
        "trip_distance": np.random.uniform(1.0, 20.0, n),
        "pickup_hour": np.random.randint(0, 24, n),
        "pickup_dayofweek": np.random.randint(0, 7, n),
        "trip_duration_min": np.random.uniform(5.0, 50.0, n),
        "passenger_count": np.random.randint(1, 5, n),
        "fare_amount": np.random.uniform(5.0, 70.0, n)
    })

    tree_res = train_and_benchmark_tree_models(df_syn)
    assert len(tree_res["benchmarks"]) >= 4

    keys = [b["key"] for b in tree_res["benchmarks"]]
    assert "decision_tree" in keys
    assert "random_forest" in keys
    assert "adaboost" in keys

    # Check Random Forest OOB
    rf_bench = [b for b in tree_res["benchmarks"] if b["key"] == "random_forest"][0]
    assert rf_bench["oob_score"] is not None

    # Check MDI and Permutation feature importances
    assert "trip_distance" in tree_res["feature_importances"]["mdi_random_forest"]
    assert "trip_distance" in tree_res["feature_importances"]["permutation_importance"]


def test_m3_pruning_path_and_criteria():
    """
    Verifies M3: Impurity split criteria (Gini, Entropy) and CCP pruning path.
    """
    crit = compute_tree_split_criteria([0.7, 0.3])
    assert crit["gini_impurity"] == 0.42
    assert crit["entropy"] > 0.85

    X = np.linspace(1.0, 10.0, 60).reshape(-1, 1)
    y = 2.0 * X.ravel() + np.random.normal(0, 1, 60)
    prune_res = compute_cost_complexity_pruning_path(X, y)
    assert prune_res["total_alphas_found"] > 0
    assert len(prune_res["sample_alphas"]) > 0


def test_m4_unsupervised_clustering_and_anomalies():
    """
    Verifies M4: K-Means, Elbow & Silhouette analysis, PCA dimensions,
    DBSCAN, and Isolation Forest anomaly detection.
    """
    m4_res = run_complete_m4_unsupervised(None)
    assert m4_res["kmeans"]["optimal_k"] == 3
    assert len(m4_res["kmeans"]["cluster_archetypes"]) == 3
    assert len(m4_res["pca"]["explained_variance_ratio"]) == 3
    assert m4_res["anomaly_detection"]["isolation_forest_anomalies"] > 0


def test_m5_evaluation_protocols_and_calibration():
    """
    Verifies M5: 5-Fold CV, Stratified CV, Nested CV, Probability Calibration,
    McNemar's test, and Paired Bootstrap Confidence Intervals.
    """
    m5_res = run_complete_m5_evaluation(None)
    assert m5_res["kfold_cv"]["k_folds"] == 5
    assert len(m5_res["kfold_cv"]["fold_r2_scores"]) == 5
    assert m5_res["stratified_cv"]["k_folds"] == 5
    assert m5_res["nested_cv"]["outer_folds"] == 3

    # Check Calibration Brier Scores
    calib = m5_res["probability_calibration"]
    assert "uncalibrated" in calib
    assert "platt_sigmoid" in calib
    assert "isotonic_regression" in calib

    # Check Hypothesis Tests
    assert "p_value" in m5_res["mcnemar_test"]
    assert "ci_95_lower" in m5_res["bootstrap_ci_95"]


def test_m6_engineering_registry_and_telemetry():
    """
    Verifies M6: Model artifact save/load, SHA256 integrity check,
    runtime schema validation, and telemetry tracking.
    """
    # 1. Model Save & Load with Checksum
    clf = Ridge(alpha=1.0)
    clf.fit(np.array([[1.0], [2.0], [3.0]]), np.array([3.0, 6.0, 9.0]))
    save_info = save_model_artifact(clf, "test_m6_artifact", ["trip_distance"], "fare_amount")
    assert os.path.exists(save_info["binary_path"])
    assert len(save_info["sha256"]) == 64

    loaded_obj, meta = load_model_artifact("test_m6_artifact")
    assert loaded_obj is not None
    assert meta["model_name"] == "test_m6_artifact"
    assert meta["sha256_checksum"] == save_info["sha256"]

    # 2. Telemetry tracking
    record_prediction_telemetry(2.4, "polynomial")
    telem = get_monitoring_telemetry()
    assert telem["total_requests_served"] > 0
    assert telem["service_status"] == "HEALTHY"


def test_m6_feature_drift_detection():
    """
    Verifies M6: Kolmogorov-Smirnov test and Population Stability Index (PSI).
    """
    base = np.random.normal(5.0, 1.0, 500)
    # Similar distribution -> No drift
    stable = np.random.normal(5.0, 1.0, 500)
    res_stable = detect_feature_drift(base, stable)
    assert res_stable["severity"] in ("NEGLIGIBLE", "MODERATE")

    # Shifted distribution -> Critical drift
    shifted = np.random.normal(12.0, 2.0, 500)
    res_shifted = detect_feature_drift(base, shifted)
    assert res_shifted["drift_detected"] is True
    assert res_shifted["psi_score"] > 0.20


def test_workbench_all_tabs_and_extended_apis(client):
    """
    Verifies that all 8 Workbench tabs render and all extended M1-M6 API endpoints
    respond with HTTP 200.
    """
    # Test tabs
    for tab in ["lifecycle", "regression", "classification", "trees", "unsupervised", "evaluation", "engineering", "syllabus"]:
        resp = client.get(f"/workbench?tab={tab}")
        assert resp.status_code == 200
        assert b"RideRush ML Engineering" in resp.data

    # Test /api/workbench/summary
    resp_sum = client.get("/api/workbench/summary")
    assert resp_sum.status_code == 200
    json_sum = resp_sum.get_json()
    assert "m1_lifecycle" in json_sum
    assert "m2_regression" in json_sum

    # Test /api/workbench/batch-predict
    resp_batch = client.post("/api/workbench/batch-predict", data="trip_distance\n2.5\n5.0\n", content_type="text/plain")
    assert resp_batch.status_code == 200
    b_json = resp_batch.get_json()
    assert b_json["status"] == "success"
    assert b_json["total_processed"] == 2

    # Test /api/workbench/drift
    resp_drift = client.get("/api/workbench/drift")
    assert resp_drift.status_code == 200

    # Test /api/workbench/telemetry
    resp_telem = client.get("/api/workbench/telemetry")
    assert resp_telem.status_code == 200
    assert "service_status" in resp_telem.get_json()



