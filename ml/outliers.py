"""
Outlier Analysis & Treatment Module (CO1 Syllabus) for NYC Taxi Big Data.

Provides modular functions for IQR calculations (Q1, Q2/Median, Q3, IQR, Lower/Upper Bounds),
Boxplot visualization generation, Outlier Removal impact evaluation, Retain Outliers baseline,
Winsorization capping, and multi-treatment comparative analysis using Dask and Pandas.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import dask.dataframe as dd

CHARTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "charts")
TARGET_NUMERICAL_COLS = ['trip_distance', 'trip_duration_minutes', 'fare_amount', 'total_amount']


def calculate_iqr_statistics(df_clean):
    """
    Computes Q1 (25%), Q2/Median (50%), Q3 (75%), IQR, Lower Bound, Upper Bound,
    Outlier Count, and Outlier Percentage for numerical features using Dask aggregations.
    """
    total_clean_rows = len(df_clean)
    iqr_results = []

    for col in TARGET_NUMERICAL_COLS:
        q25, q50, q75 = df_clean[col].quantile([0.25, 0.50, 0.75]).compute().values
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr

        outlier_cnt = int(((df_clean[col] < lower) | (df_clean[col] > upper)).sum().compute())
        outlier_pct = round((outlier_cnt / total_clean_rows) * 100.0, 2)

        iqr_results.append({
            "variable": col,
            "q1": round(float(q25), 2),
            "median": round(float(q50), 2),
            "q3": round(float(q75), 2),
            "iqr": round(float(iqr), 2),
            "lower_bound": round(float(lower), 2),
            "upper_bound": round(float(upper), 2),
            "potential_outliers": outlier_cnt,
            "outlier_pct": outlier_pct
        })

    return iqr_results


def create_boxplots(df_clean):
    """
    Generates a 2x2 subplot boxplot chart figure saved as static/charts/outlier_boxplots.png
    using a representative 5% sample to avoid memory overflow.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    sns.set_theme(style="darkgrid")

    # Sample 5% clean dataset for boxplot visualization
    sample_df = df_clean.sample(frac=0.05, random_state=42).compute()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Numerical Feature Outlier Boxplots (IQR Method)", fontsize=14, fontweight='bold', y=0.98)

    plots_config = [
        ('trip_distance', 'Trip Distance (miles)', axes[0, 0], '#f0883e', 30),
        ('trip_duration_minutes', 'Trip Duration (minutes)', axes[0, 1], '#3fb950', 90),
        ('fare_amount', 'Fare Amount ($)', axes[1, 0], '#a371f7', 100),
        ('total_amount', 'Total Amount ($)', axes[1, 1], '#58a6ff', 120)
    ]

    for col, label, ax, color, limit in plots_config:
        data_to_plot = sample_df[sample_df[col] <= limit][col]
        sns.boxplot(x=data_to_plot, ax=ax, color=color, fliersize=3, boxprops=dict(alpha=0.85))
        ax.set_title(f"Boxplot: {label}", fontsize=11, fontweight='bold')
        ax.set_xlabel(label)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(CHARTS_DIR, "outlier_boxplots.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return plot_path


def evaluate_outlier_treatments(df_clean, iqr_table):
    """
    Evaluates and compares the 3 outlier treatment strategies:
    1. Remove Outliers (IQR Filtering)
    2. Retain Outliers (Base Clean Dataset)
    3. Winsorization (Lower & Upper Bound Capping)
    """
    total_clean_rows = len(df_clean)

    # 1. Remove Outliers (Analytical Evaluation)
    # Build combined Dask condition across target numerical columns
    condition = True
    for row in iqr_table:
        col = row["variable"]
        lower = row["lower_bound"]
        upper = row["upper_bound"]
        condition = condition & (df_clean[col] >= lower) & (df_clean[col] <= upper)

    rows_after_removal = int(df_clean[condition].shape[0].compute())
    removed_cnt = total_clean_rows - rows_after_removal
    retention_pct = round((rows_after_removal / total_clean_rows) * 100.0, 2)

    remove_impact = {
        "original_clean_records": total_clean_rows,
        "records_after_removal": rows_after_removal,
        "records_removed": removed_cnt,
        "retention_pct": retention_pct
    }

    # 2. Retain Outliers (Baseline)
    retain_impact = {
        "original_clean_records": total_clean_rows,
        "records_retained": total_clean_rows,
        "records_removed": 0,
        "retention_pct": 100.0
    }

    # 3. Winsorization Comparison (Using 5% sample for fast before/after stat comparison)
    sample_df = df_clean.sample(frac=0.05, random_state=42).compute()
    winsorize_comparison = []

    for row in iqr_table:
        col = row["variable"]
        lower = row["lower_bound"]
        upper = row["upper_bound"]

        orig_series = sample_df[col]
        capped_series = np.clip(orig_series, lower, upper)

        winsorize_comparison.append({
            "variable": col,
            "mean_before": round(float(orig_series.mean()), 2),
            "mean_after": round(float(capped_series.mean()), 2),
            "median_before": round(float(orig_series.median()), 2),
            "median_after": round(float(capped_series.median()), 2),
            "std_before": round(float(orig_series.std()), 2),
            "std_after": round(float(capped_series.std()), 2)
        })

    # Summary table comparing all 3 approaches
    method_comparison_table = [
        {
            "method": "Remove Outliers (IQR Filtering)",
            "records_removed": f"{removed_cnt:,}",
            "data_preserved_pct": f"{retention_pct}%",
            "effect": "Eliminates all records outside Q1 - 1.5*IQR and Q3 + 1.5*IQR bounds"
        },
        {
            "method": "Retain Outliers (Keep Original)",
            "records_removed": "0",
            "data_preserved_pct": "100.0%",
            "effect": "Preserves all genuine extreme trips (e.g. long borough or airport travel)"
        },
        {
            "method": "Winsorization (Bound Capping)",
            "records_removed": "0",
            "data_preserved_pct": "100.0%",
            "effect": "Caps extreme values at Lower & Upper bounds without dropping any rows"
        }
    ]

    return {
        "remove_impact": remove_impact,
        "retain_impact": retain_impact,
        "winsorize_comparison": winsorize_comparison,
        "method_comparison_table": method_comparison_table
    }


def run_complete_outlier_analysis(df_clean):
    """
    Master runner function for CO1 Outlier Analysis.
    Calculates IQR metrics, creates boxplots, and evaluates treatment strategies.
    """
    iqr_table = calculate_iqr_statistics(df_clean)
    boxplots_path = create_boxplots(df_clean)
    treatments = evaluate_outlier_treatments(df_clean, iqr_table)

    return {
        "iqr_table": iqr_table,
        "boxplots_chart": "static/charts/outlier_boxplots.png",
        "treatments": treatments
    }
