"""
Flask Web Application for NYC Taxi Big Data EDA & Machine Learning.

Renders modern dark-themed interactive web pages for dataset understanding,
data quality/cleaning audits, EDA visualizations, calculated key insights,
CO1 outlier analysis, CO1 missing value analysis, and Simple Linear Regression with Gradient Descent trip fare predictions.
"""

import os
import socket
from flask import Flask, render_template, request, jsonify
from data import (
    get_data, predict_fare, predict_workbench_fare,
    execute_batch_inference, get_monitoring_dashboard_data
)

app = Flask(__name__)


@app.route("/")
def home():
    """
    Home Dashboard page.
    """
    try:
        pipeline_data = get_data()
        return render_template("index.html", data=pipeline_data)
    except Exception as e:
        return render_template("index.html", error=str(e))


@app.route("/data")
def data_page():
    """
    Phase 2: Data Understanding page.
    """
    try:
        pipeline_data = get_data()
        return render_template("data_loading.html", info=pipeline_data["dataset_info"])
    except Exception as e:
        return render_template("data_loading.html", error=str(e))


@app.route("/cleaning")
def cleaning_page():
    """
    Phase 3 & 4: Data Quality Analysis & Data Cleaning page.
    """
    try:
        pipeline_data = get_data()
        return render_template(
            "cleaning.html",
            quality=pipeline_data["quality_metrics"],
            stats=pipeline_data["cleaning_stats"]
        )
    except Exception as e:
        return render_template("cleaning.html", error=str(e))


@app.route("/eda")
def eda_page():
    """
    Phase 5: Complete EDA Dashboard, Outlier Audit & Missing Value Analysis.
    """
    try:
        pipeline_data = get_data()
        return render_template(
            "eda.html",
            outliers=pipeline_data["outlier_table"],
            outlier_data=pipeline_data.get("outliers", {}),
            missing_value_data=pipeline_data.get("missing_values", {}),
            scaling_data=pipeline_data.get("scaling", {}),
            duplicate_data=pipeline_data.get("duplicates", {}),
            insights=pipeline_data["insights"]
        )
    except Exception as e:
        return render_template("eda.html", error=str(e))


@app.route("/insights")
def insights_page():
    """
    EDA Key Insights page displaying calculated metrics only.
    """
    try:
        pipeline_data = get_data()
        return render_template("insights.html", insights=pipeline_data["insights"])
    except Exception as e:
        return render_template("insights.html", error=str(e))


@app.route("/prediction", methods=["GET", "POST"])
def prediction_page():
    """
    Simple Linear Regression & Gradient Descent Fare Predictor page.
    """
    try:
        pipeline_data = get_data()
        dist_input = 2.5
        if request.method == "POST":
            dist_input = request.form.get("distance", 2.5)
        elif request.args.get("distance"):
            dist_input = request.args.get("distance")

        pred_result = predict_fare(dist_input)
        return render_template(
            "prediction.html",
            reg=pipeline_data.get("regression", {}),
            prediction=pred_result
        )
    except Exception as e:
        return render_template("prediction.html", error=str(e), reg={}, prediction={"distance": 2.5, "predicted_fare": 0.0, "formula": ""})


@app.route("/api/predict")
def api_predict():
    """
    JSON API endpoint for fare prediction based on trip distance.
    Usage: GET /api/predict?distance=5.2
    """
    try:
        dist_arg = request.args.get("distance", 2.5)
        result = predict_fare(dist_arg)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/summary")
def api_summary():
    """
    API endpoint returning raw pipeline JSON summary.
    """
    try:
        pipeline_data = get_data()
        return jsonify(pipeline_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/workbench", methods=["GET", "POST"])
def workbench_page():
    """
    Comprehensive ML Workbench page covering M1-M6 syllabus:
    Lifecycle, Regression, Classification, Tree Ensembles,
    Unsupervised Learning, Evaluation & Calibration, and ML Engineering.
    """
    try:
        pipeline_data = get_data()
        workbench_data = pipeline_data.get("workbench", {})

        dist_input = 5.0
        model_key = "baseline_gd"
        batch_res = None
        tab_active = request.args.get("tab", "regression")

        if request.method == "POST":
            form_type = request.form.get("form_type", "single")
            if form_type == "batch":
                tab_active = "lifecycle"
                csv_payload = request.form.get("batch_csv", "")
                batch_model = request.form.get("batch_model_key", "polynomial")
                if csv_payload.strip():
                    try:
                        batch_res = execute_batch_inference(csv_payload, batch_model)
                    except Exception as b_err:
                        batch_res = {"status": "error", "message": str(b_err)}
            else:
                dist_input = request.form.get("distance", 5.0)
                model_key = request.form.get("model_key", "baseline_gd")
        elif request.args.get("distance"):
            dist_input = request.args.get("distance", 5.0)
            if request.args.get("model_key"):
                model_key = request.args.get("model_key")

        pred_result = predict_workbench_fare(dist_input, model_key)
        telemetry_data = get_monitoring_dashboard_data()

        return render_template(
            "workbench.html",
            workbench=workbench_data,
            m1=pipeline_data.get("m1_lifecycle", {}),
            m2_clf=pipeline_data.get("m2_classification", {}),
            m3_trees=pipeline_data.get("m3_trees", {}),
            m4_unsup=pipeline_data.get("m4_unsupervised", {}),
            m5_eval=pipeline_data.get("m5_evaluation", {}),
            m6_eng=pipeline_data.get("m6_engineering", {}),
            prediction=pred_result,
            batch_result=batch_res,
            telemetry=telemetry_data,
            active_tab=tab_active
        )
    except Exception as e:
        return render_template("workbench.html", error=str(e), workbench={}, prediction=None)


@app.route("/api/workbench")
def api_workbench():
    """
    API endpoint returning regression workbench benchmarking data and metrics.
    """
    try:
        pipeline_data = get_data()
        workbench_data = pipeline_data.get("workbench", {})
        return jsonify(workbench_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/workbench/summary")
def api_workbench_summary():
    """
    Comprehensive JSON API endpoint returning all M1-M6 model metrics and schemas.
    """
    try:
        pipeline_data = get_data()
        summary = {
            "m1_lifecycle": pipeline_data.get("m1_lifecycle", {}),
            "m2_regression": pipeline_data.get("workbench", {}),
            "m2_classification": pipeline_data.get("m2_classification", {}),
            "m3_trees": pipeline_data.get("m3_trees", {}),
            "m4_unsupervised": pipeline_data.get("m4_unsupervised", {}),
            "m5_evaluation": pipeline_data.get("m5_evaluation", {}),
            "m6_engineering": pipeline_data.get("m6_engineering", {})
        }
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/workbench/predict")
def api_workbench_predict():
    """
    JSON API endpoint for multi-model fare prediction with schema validation.
    Usage: GET /api/workbench/predict?distance=5.2&model=ridge
    """
    try:
        dist_arg = request.args.get("distance", 5.0)
        model_arg = request.args.get("model", "baseline_gd")
        result = predict_workbench_fare(dist_arg, model_arg)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/workbench/batch-predict", methods=["POST"])
def api_workbench_batch_predict():
    """
    Batch prediction endpoint supporting JSON tabular records or raw CSV.
    Max 1000 records per request.
    """
    try:
        if request.is_json:
            records = request.json.get("records", [])
            model_key = request.json.get("model", "polynomial")
            df_input = pd.DataFrame(records)
            result = execute_batch_inference(df_input, model_key)
        else:
            csv_data = request.data.decode("utf-8")
            model_key = request.args.get("model", "polynomial")
            result = execute_batch_inference(csv_data, model_key)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/workbench/drift")
def api_workbench_drift():
    """
    Returns feature distribution drift audit and Kolmogorov-Smirnov test.
    """
    try:
        pipeline_data = get_data()
        m6 = pipeline_data.get("m6_engineering", {})
        return jsonify(m6.get("drift_monitoring_demo", {}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/workbench/telemetry")
def api_workbench_telemetry():
    """
    Returns production serving telemetry (request counts, latency percentiles).
    """
    try:
        telemetry = get_monitoring_dashboard_data()
        return jsonify(telemetry)
    except Exception as e:
        return jsonify({"error": str(e)}), 500




def find_available_port(start_port=5050):
    """
    Finds an available TCP port starting from start_port.
    """
    port = start_port
    while port < start_port + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(('127.0.0.1', port)) != 0:
                return port
        port += 1
    return start_port


if __name__ == "__main__":
    desired_port = int(os.environ.get("PORT", 5050))
    port = find_available_port(desired_port)
    print(f"Starting server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
