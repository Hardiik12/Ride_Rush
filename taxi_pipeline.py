"""
Pipeline Orchestrator for NYC Taxi Big Data EDA, Outlier Analysis, Missing Values & Regression Web Application.

Coordinates the end-to-end workflow:
Data Loading -> Understanding -> Quality -> Cleaning -> EDA -> Outlier Analysis -> Missing Value Analysis -> Regression Model Training.
Caches structured metrics into JSON for high-performance Flask routing.
"""

import os
import json
from ml.data_loader import load_taxi_data, get_dataset_info
from ml.data_cleaning import check_data_quality, clean_taxi_data
from ml.eda import run_full_eda
from ml.outliers import run_complete_outlier_analysis
from ml.missing_values import run_complete_missing_value_analysis
from ml.scaling import run_complete_scaling_analysis
from ml.duplicates import run_complete_duplicate_analysis
from ml.linear_regression import train_and_save_regression_model
from ml.regression_workbench import run_complete_regression_workbench
from ml.common import extract_representative_ml_sample
from ml.m1_lifecycle import run_complete_m1_lifecycle
from ml.m2_classification import run_complete_m2_classification
from ml.m3_trees import run_complete_m3_trees
from ml.m4_unsupervised import run_complete_m4_unsupervised
from ml.m5_evaluation import run_complete_m5_evaluation
from ml.m6_engineering import run_complete_m6_engineering

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "reports")
CACHE_FILE = os.path.join(REPORTS_DIR, "pipeline_summary.json")


class TaxiDataPipeline:
    def __init__(self, data_path=None):
        self.data_path = data_path or os.path.join(os.path.dirname(__file__), "data", "yellow_tripdata_2025-01.parquet")
        self.results = {}

    def run_pipeline(self, force=False):
        """
        Executes full pipeline if not cached or force=True.
        """
        os.makedirs(REPORTS_DIR, exist_ok=True)
        if not force and os.path.exists(CACHE_FILE):
            print(f"[INFO] Loading cached pipeline summary from {CACHE_FILE}")
            with open(CACHE_FILE, "r") as f:
                self.results = json.load(f)
            return self.results

        print("=" * 60)
        print("STARTING NYC TAXI BIG DATA PIPELINE (DASK + CO1 ANALYSIS + GRADIENT DESCENT)")
        print("=" * 60)

        # 1. Data Loading & Understanding
        print("[STEP 1] Loading Parquet Dataset lazily...")
        df_raw = load_taxi_data(self.data_path)
        print("[STEP 2] Extracting Dataset Information...")
        info = get_dataset_info(df_raw, self.data_path)

        # 2. Data Quality Analysis
        print("[STEP 3] Running Data Quality Checks...")
        quality = check_data_quality(df_raw)

        # 3. Data Cleaning
        print("[STEP 4] Executing Dask Data Cleaning Filters...")
        df_clean, cleaning_stats = clean_taxi_data(df_raw)

        # 4. EDA & Visualization
        print("[STEP 5] Generating EDA Visualizations & Insights...")
        eda_results = run_full_eda(df_clean)

        # 5. CO1 Outlier Analysis & Treatment Comparisons
        print("[STEP 5b] Executing CO1 Outlier Analysis (Boxplots, IQR, Winsorization)...")
        outlier_results = run_complete_outlier_analysis(df_clean)

        # 6. CO1 Missing Value Analysis & Imputation Techniques
        print("[STEP 5c] Executing CO1 Missing Value Analysis & Imputation Techniques...")
        missing_value_results = run_complete_missing_value_analysis(df_clean)

        # 7. CO1 Feature Scaling & Leakage Analysis
        print("[STEP 5d] Executing CO1 Feature Scaling & Leakage Safeguards Analysis...")
        scaling_results = run_complete_scaling_analysis(df_clean)

        # 8. CO1 Duplicate Detection & Treatment Analysis
        print("[STEP 5e] Executing CO1 Duplicate Detection & Treatment Evaluation...")
        duplicate_results = run_complete_duplicate_analysis(df_clean)

        # 9. Simple Linear Regression & Gradient Descent
        print("[STEP 6] Training Simple Linear Regression Model with Gradient Descent...")
        regression_results = train_and_save_regression_model(df_clean)

        # 10. M2 Regression Model Workbench
        print("[STEP 7] Executing M2 Regression Model Workbench (Polynomial, Ridge, Lasso, Elastic Net)...")
        workbench_results = run_complete_regression_workbench(df_clean)

        # 11. M1-M6 Extended Modules
        print("[STEP 8] Sampling representative feature-engineered data for M1-M6 modules...")
        sample_df = extract_representative_ml_sample(df_clean, sample_frac=0.03, random_state=42)

        print("[STEP 9] Executing M1 Lifecycle Pipeline & Batch Engine...")
        m1_results = run_complete_m1_lifecycle(df_clean)

        print("[STEP 10] Executing M2 Supervised Classification Workbench...")
        m2_clf_results = run_complete_m2_classification(df_clean)

        print("[STEP 11] Executing M3 Tree Models & Boosting Benchmarks...")
        m3_tree_results = run_complete_m3_trees(df_clean)

        print("[STEP 12] Executing M4 Unsupervised Learning & Clustering...")
        m4_unsup_results = run_complete_m4_unsupervised(df_clean)

        print("[STEP 13] Executing M5 Evaluation, Model Selection & Calibration...")
        m5_eval_results = run_complete_m5_evaluation(df_clean)

        print("[STEP 14] Executing M6 ML Engineering & Registry Operations...")
        m6_eng_results = run_complete_m6_engineering(sample_df)

        self.results = {
            "dataset_info": info,
            "quality_metrics": quality,
            "cleaning_stats": cleaning_stats,
            "outlier_table": eda_results["outlier_table"],
            "insights": eda_results["insights"],
            "outliers": outlier_results,
            "missing_values": missing_value_results,
            "scaling": scaling_results,
            "duplicates": duplicate_results,
            "regression": regression_results,
            "workbench": workbench_results,
            "m1_lifecycle": m1_results,
            "m2_classification": m2_clf_results,
            "m3_trees": m3_tree_results,
            "m4_unsupervised": m4_unsup_results,
            "m5_evaluation": m5_eval_results,
            "m6_engineering": m6_eng_results
        }

        # Cache results to JSON safely
        with open(CACHE_FILE, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print("[SUCCESS] Pipeline completed successfully! Cache saved to:", CACHE_FILE)
        print("=" * 60)
        return self.results


if __name__ == "__main__":
    pipeline = TaxiDataPipeline()
    pipeline.run_pipeline(force=True)
