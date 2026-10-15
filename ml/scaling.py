"""
Feature Scaling & Normalization Module (CO1 Syllabus) for NYC Taxi Big Data.

Provides modular implementations for:
1. Standardization / Z-score (StandardScaler: mean=0, std=1)
2. Min-Max Scaling (MinMaxScaler: bound to [0, 1])
3. Robust Scaling (RobustScaler: scaled by median & IQR)
4. Before/after distribution statistics (mean, std, min, median, max, IQR)
5. Strict Data Leakage Prevention (fitting strictly on training set only)
6. Model Selection Guide (models requiring scaling vs tree models invariant to scaling)
7. Comparative visualization generation saved to static/charts/scaling_comparison.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

CHARTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "charts")
TARGET_SCALING_COLS = ['trip_distance', 'fare_amount', 'trip_duration_minutes', 'total_amount']


def compute_distribution_stats(series, name=""):
    """
    Computes distribution summary statistics for a 1D numeric array or series.
    Returns: dict with mean, std, min, median, max, iqr, q25, q75.
    """
    arr = np.asarray(series).ravel()
    q25, median, q75 = np.percentile(arr, [25, 50, 75])
    iqr = q75 - q25
    return {
        "feature": name,
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "q25": round(float(q25), 4),
        "median": round(float(median), 4),
        "q75": round(float(q75), 4),
        "max": round(float(np.max(arr)), 4),
        "iqr": round(float(iqr), 4)
    }


def standard_scaling(train_df, test_df=None, feature_cols=None):
    """
    Standardizes numerical features using Z-score formula: z = (x - mean) / std.
    Fits the StandardScaler strictly on train_df, then transforms train_df and test_df.
    
    Returns:
        dict containing:
        - 'scaler_name': 'StandardScaler (Z-score)'
        - 'formula': 'z = (x - mean) / std'
        - 'fitted_parameters': dict of mean and scale per feature
        - 'train_stats_before': list of stats before scaling
        - 'train_stats_after': list of stats after scaling (mean~0, std~1)
        - 'test_stats_after': list of test stats if test_df provided
        - 'train_transformed': pd.DataFrame of scaled training features
        - 'test_transformed': pd.DataFrame of scaled testing features (if test_df given)
    """
    if feature_cols is None:
        feature_cols = [c for c in TARGET_SCALING_COLS if c in train_df.columns]

    scaler = StandardScaler()
    X_train = train_df[feature_cols].copy()
    scaler.fit(X_train)

    train_scaled = pd.DataFrame(scaler.transform(X_train), columns=feature_cols, index=train_df.index)

    stats_before = [compute_distribution_stats(X_train[c], c) for c in feature_cols]
    stats_after = [compute_distribution_stats(train_scaled[c], c) for c in feature_cols]

    fitted_params = {
        col: {"mean": round(float(m), 4), "std": round(float(s), 4)}
        for col, m, s in zip(feature_cols, scaler.mean_, scaler.scale_)
    }

    test_scaled = None
    test_stats_after = None
    if test_df is not None:
        X_test = test_df[feature_cols].copy()
        test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=test_df.index)
        test_stats_after = [compute_distribution_stats(test_scaled[c], c) for c in feature_cols]

    return {
        "scaler_name": "StandardScaler (Z-Score)",
        "formula": "z = (x - μ) / σ",
        "description": "Transforms features to have zero mean (μ=0) and unit standard deviation (σ=1).",
        "outlier_sensitivity": "High — Outliers skew the mean and expand standard deviation, compressing non-outliers into a narrow band.",
        "fitted_parameters": fitted_params,
        "stats_before": stats_before,
        "stats_after": stats_after,
        "test_stats_after": test_stats_after,
        "train_scaled_df": train_scaled,
        "test_scaled_df": test_scaled
    }


def minmax_scaling(train_df, test_df=None, feature_cols=None, feature_range=(0, 1)):
    """
    Performs Min-Max normalization: x_norm = (x - min) / (max - min).
    Fits MinMaxScaler strictly on train_df, transforms train_df and test_df.
    """
    if feature_cols is None:
        feature_cols = [c for c in TARGET_SCALING_COLS if c in train_df.columns]

    scaler = MinMaxScaler(feature_range=feature_range)
    X_train = train_df[feature_cols].copy()
    scaler.fit(X_train)

    train_scaled = pd.DataFrame(scaler.transform(X_train), columns=feature_cols, index=train_df.index)

    stats_before = [compute_distribution_stats(X_train[c], c) for c in feature_cols]
    stats_after = [compute_distribution_stats(train_scaled[c], c) for c in feature_cols]

    fitted_params = {
        col: {"min": round(float(mn), 4), "max": round(float(mx), 4), "range": round(float(mx - mn), 4)}
        for col, mn, mx in zip(feature_cols, scaler.data_min_, scaler.data_max_)
    }

    test_scaled = None
    test_stats_after = None
    if test_df is not None:
        X_test = test_df[feature_cols].copy()
        test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=test_df.index)
        test_stats_after = [compute_distribution_stats(test_scaled[c], c) for c in feature_cols]

    return {
        "scaler_name": "MinMaxScaler",
        "formula": "x' = (x - x_min) / (x_max - x_min)",
        "description": f"Compresses features strictly into the bounded range [{feature_range[0]}, {feature_range[1]}].",
        "outlier_sensitivity": "Very High — Extreme outliers set x_max or x_min, squashing the vast majority of regular data into a minuscule sub-interval.",
        "fitted_parameters": fitted_params,
        "stats_before": stats_before,
        "stats_after": stats_after,
        "test_stats_after": test_stats_after,
        "train_scaled_df": train_scaled,
        "test_scaled_df": test_scaled
    }


def robust_scaling(train_df, test_df=None, feature_cols=None):
    """
    Scales features using median and Interquartile Range (IQR):
    x_robust = (x - median) / IQR.
    Fits RobustScaler strictly on train_df, transforms train_df and test_df.
    """
    if feature_cols is None:
        feature_cols = [c for c in TARGET_SCALING_COLS if c in train_df.columns]

    scaler = RobustScaler()
    X_train = train_df[feature_cols].copy()
    scaler.fit(X_train)

    train_scaled = pd.DataFrame(scaler.transform(X_train), columns=feature_cols, index=train_df.index)

    stats_before = [compute_distribution_stats(X_train[c], c) for c in feature_cols]
    stats_after = [compute_distribution_stats(train_scaled[c], c) for c in feature_cols]

    fitted_params = {
        col: {"center_median": round(float(c), 4), "scale_iqr": round(float(s), 4)}
        for col, c, s in zip(feature_cols, scaler.center_, scaler.scale_)
    }

    test_scaled = None
    test_stats_after = None
    if test_df is not None:
        X_test = test_df[feature_cols].copy()
        test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=test_df.index)
        test_stats_after = [compute_distribution_stats(test_scaled[c], c) for c in feature_cols]

    return {
        "scaler_name": "RobustScaler",
        "formula": "x' = (x - Median) / (Q3 - Q1)",
        "description": "Centers data around the median (median=0) and divides by IQR (IQR=1). Does not cap bounds.",
        "outlier_sensitivity": "Low / Robust — Outliers do not distort the median or the middle 50% spread, preserving distribution shape without collapsing typical records.",
        "fitted_parameters": fitted_params,
        "stats_before": stats_before,
        "stats_after": stats_after,
        "test_stats_after": test_stats_after,
        "train_scaled_df": train_scaled,
        "test_scaled_df": test_scaled
    }


def demonstrate_scaling_leakage(sample_df, feature='fare_amount', test_size=0.2, random_state=42):
    """
    Demonstrates the difference between:
    1. LEAKY Scaling (fitting scaler on train + test combined before split)
    2. SAFE Scaling (fitting scaler strictly on train set, then transforming test)
    
    Returns metrics showing the subtle parameter shift and explaining data leakage consequences.
    """
    train_df, test_df = train_test_split(sample_df, test_size=test_size, random_state=random_state)

    # SAFE Approach: fit ONLY on train
    safe_scaler = StandardScaler()
    safe_scaler.fit(train_df[[feature]])
    test_safe_transformed = safe_scaler.transform(test_df[[feature]]).ravel()

    # LEAKY Approach: fit on FULL dataset (train + test)
    leaky_scaler = StandardScaler()
    leaky_scaler.fit(sample_df[[feature]])
    test_leaky_transformed = leaky_scaler.transform(test_df[[feature]]).ravel()

    return {
        "feature_demonstrated": feature,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "safe_train_mean": round(float(safe_scaler.mean_[0]), 4),
        "safe_train_std": round(float(safe_scaler.scale_[0]), 4),
        "leaky_combined_mean": round(float(leaky_scaler.mean_[0]), 4),
        "leaky_combined_std": round(float(leaky_scaler.scale_[0]), 4),
        "test_mean_with_safe_fit": round(float(np.mean(test_safe_transformed)), 4),
        "test_mean_with_leaky_fit": round(float(np.mean(test_leaky_transformed)), 4),
        "test_std_with_safe_fit": round(float(np.std(test_safe_transformed)), 4),
        "test_std_with_leaky_fit": round(float(np.std(test_leaky_transformed)), 4),
        "mean_shift": round(float(abs(safe_scaler.mean_[0] - leaky_scaler.mean_[0])), 6),
        "verdict": "Fitting the scaler on the entire dataset incorporates test-set distribution parameters (μ, σ) into training preprocessing. In production or cross-validation, this produces overly optimistic validation performance and causes pipeline drift on genuine unseen data."
    }


def get_scaler_comparison_matrix():
    """
    Returns an educational comparison table of StandardScaler, MinMaxScaler, and RobustScaler.
    """
    return [
        {
            "scaler": "StandardScaler (Z-Score)",
            "formula": "z = (x - μ) / σ",
            "center": "Mean = 0",
            "spread": "Std Dev = 1",
            "bounded": "No ([-∞, +∞])",
            "outlier_impact": "High. Outliers distort mean & variance, compressing non-outlier cluster.",
            "recommended_use": "Features following roughly Gaussian distributions; Gradient Descent algorithms; Ridge/Lasso regularization; PCA; Logistic Regression."
        },
        {
            "scaler": "MinMaxScaler",
            "formula": "x' = (x - x_min) / (x_max - x_min)",
            "center": "Depends on bounds",
            "spread": "Fixed to [0, 1]",
            "bounded": "Yes ([0, 1])",
            "outlier_impact": "Severe. Extreme outliers push x_max far out, squeezing valid data into a narrow sliver.",
            "recommended_use": "Algorithms requiring strictly bounded inputs (Image pixel values, Neural Networks with Sigmoid activations, KNN distance where bounded range is required and outliers are absent)."
        },
        {
            "scaler": "RobustScaler",
            "formula": "x' = (x - Median) / IQR",
            "center": "Median = 0",
            "spread": "IQR = 1",
            "bounded": "No ([-∞, +∞])",
            "outlier_impact": "Low / Robust. Uses 50th percentile and middle 50% range; extreme values cannot skew center or spread.",
            "recommended_use": "Heavy-tailed, skewed datasets with persistent genuine outliers (e.g., NYC Taxi trip distances, airport fares, surge rates) where removing outliers is undesirable."
        }
    ]


def get_model_scaling_guidance():
    """
    Returns educational guidance on which ML models strictly require feature scaling
    versus tree-based models that are scale-invariant.
    """
    return {
        "models_needing_scaling": [
            {
                "model_category": "Distance-Based Models",
                "examples": "KNN (K-Nearest Neighbors), K-Means Clustering, DBSCAN",
                "why_needed": "Compute Euclidean or Manhattan distances between vectors. A feature with larger magnitude (e.g. Total Fare in $ vs Passenger Count in units) completely dominates the distance metric, rendering smaller-scale features negligible."
            },
            {
                "model_category": "Gradient Descent Optimization",
                "examples": "Simple & Multiple Linear Regression (GD), Logistic Regression, Neural Networks",
                "why_needed": "Unscaled features create elongated, elliptical cost surface contours. Gradient descent oscillates wildly and requires an excessively small learning rate to converge. Scaling produces circular loss contours, accelerating convergence."
            },
            {
                "model_category": "Regularized Linear Models",
                "examples": "Ridge Regression (L2), Lasso Regression (L1), ElasticNet",
                "why_needed": "Regularization penalizes weight magnitudes (λ ∑ w_i²). If features have unequal scales, the penalty disproportionately punishes coefficients of smaller-scale features, biasing feature selection."
            },
            {
                "model_category": "Kernel & Margin-Based Models",
                "examples": "SVM (Support Vector Machines), PCA (Principal Component Analysis)",
                "why_needed": "SVM maximizes the margin in feature space; unscaled attributes distort the margin boundary. PCA identifies axes of maximum variance; unscaled high-variance features falsely capture all principal components."
            }
        ],
        "models_not_needing_scaling": [
            {
                "model_category": "Decision Trees & Ensembles",
                "examples": "DecisionTreeClassifier/Regressor, Random Forest, Extra Trees",
                "why_not_needed": "Tree algorithms split nodes feature-by-feature based on threshold inequalities (x_i <= threshold) evaluated via Gini Impurity, Entropy, or MSE reduction. Monotonic transformations do not change the ranking or split points of single features."
            },
            {
                "model_category": "Gradient Boosted Decision Trees",
                "examples": "XGBoost, LightGBM, CatBoost, GradientBoostingClassifier",
                "why_not_needed": "Like Random Forests, boosting algorithms construct ensembles of decision tree stumps/leaves. Split searching operates independently along each 1D coordinate, making them completely invariant to monotonic feature scaling."
            },
            {
                "model_category": "Naive Bayes (Certain Variants)",
                "examples": "CategoricalNB, MultinomialNB",
                "why_not_needed": "Count-based frequency probability distributions evaluate class likelihoods independently for categorical inputs."
            }
        ]
    }


def create_scaling_comparison_chart(sample_df, feature_cols=None):
    """
    Generates a 2x4 comparative subplot chart:
    Row 1: trip_distance (Original, StandardScaler, MinMaxScaler, RobustScaler)
    Row 2: fare_amount (Original, StandardScaler, MinMaxScaler, RobustScaler)
    Saves figure to static/charts/scaling_comparison.png.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    if feature_cols is None:
        feature_cols = ['trip_distance', 'fare_amount']

    # Sample for chart rendering (up to 20,000 points to keep chart crisp & fast)
    data = sample_df[feature_cols].copy()
    if len(data) > 20000:
        data = data.sample(n=20000, random_state=42)

    # Fit scalers
    std_s = StandardScaler().fit_transform(data)
    minmax_s = MinMaxScaler().fit_transform(data)
    robust_s = RobustScaler().fit_transform(data)

    std_df = pd.DataFrame(std_s, columns=feature_cols)
    minmax_df = pd.DataFrame(minmax_s, columns=feature_cols)
    robust_df = pd.DataFrame(robust_s, columns=feature_cols)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(len(feature_cols), 4, figsize=(18, 8))
    fig.patch.set_facecolor('#0d1117')

    colors = ['#58a6ff', '#3fb950', '#d29922', '#bc8cff']
    scaler_names = ['Original (Unscaled)', 'StandardScaler (Z-Score)', 'MinMaxScaler [0,1]', 'RobustScaler (Median/IQR)']

    datasets = [data, std_df, minmax_df, robust_df]

    for row_idx, col_name in enumerate(feature_cols):
        for col_idx, (df_curr, s_name, color) in enumerate(zip(datasets, scaler_names, colors)):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('#161b22')

            vals = df_curr[col_name].dropna()
            # Trim extreme visual display tail at 99.5% for original to ensure legible histogram
            if col_idx == 0:
                p99 = np.percentile(vals, 99.5)
                vals_plot = vals[vals <= p99]
            else:
                vals_plot = vals

            sns.histplot(vals_plot, kde=True, ax=ax, color=color, bins=35, stat="density", alpha=0.55)
            ax.set_title(f"{col_name}\n{s_name}", fontsize=11, color='#e6edf3', pad=8)
            ax.set_xlabel(f"Scaled Value", fontsize=9, color='#8b949e')
            ax.set_ylabel("Density", fontsize=9, color='#8b949e')
            ax.tick_params(colors='#8b949e', labelsize=8)
            ax.grid(True, linestyle='--', alpha=0.2, color='#30363d')

            # Annotate key stats
            mean_v = np.mean(vals)
            med_v = np.median(vals)
            ax.text(0.95, 0.90, f"Mean: {mean_v:.2f}\nMed: {med_v:.2f}",
                    transform=ax.transAxes, fontsize=8, color='#c9d1d9',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117', alpha=0.7, edgecolor='#30363d'))

    plt.suptitle("CO1 Feature Scaling Comparison: Effect on Skewed NYC Taxi Distributions",
                 fontsize=14, fontweight='bold', color='#f0f6fc', y=0.99)
    plt.tight_layout()

    out_path = os.path.join(CHARTS_DIR, "scaling_comparison.png")
    plt.savefig(out_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved feature scaling comparison chart to {out_path}")
    return "scaling_comparison.png"


def run_complete_scaling_analysis(df_clean, sample_frac=0.05, random_state=42):
    """
    Executes complete CO1 Feature Scaling demonstration using a representative Dask sample:
    1. Splits representative sample into train/test (80/20) to enforce strict isolation.
    2. Fits StandardScaler, MinMaxScaler, and RobustScaler on train set only.
    3. Transforms both train and test sets without data leakage.
    4. Computes distribution statistics before and after.
    5. Demonstrates data leakage effect.
    6. Generates comparison chart static/charts/scaling_comparison.png.
    7. Compiles method comparison matrix and model selection guidance.
    
    Returns:
        dict: Complete scaling summary ready for pipeline cache and Flask template.
    """
    print("  [Scaling] Sampling representative data (5%) for scaling demonstrations...")
    sample_df = df_clean[TARGET_SCALING_COLS].sample(frac=sample_frac, random_state=random_state).compute()
    sample_size = len(sample_df)

    # Strict Train/Test Split (80/20)
    train_df, test_df = train_test_split(sample_df, test_size=0.2, random_state=random_state)
    train_size = len(train_df)
    test_size = len(test_df)

    print(f"  [Scaling] Running StandardScaler on {train_size} train records...")
    std_res = standard_scaling(train_df, test_df, TARGET_SCALING_COLS)

    print(f"  [Scaling] Running MinMaxScaler on {train_size} train records...")
    minmax_res = minmax_scaling(train_df, test_df, TARGET_SCALING_COLS)

    print(f"  [Scaling] Running RobustScaler on {train_size} train records...")
    robust_res = robust_scaling(train_df, test_df, TARGET_SCALING_COLS)

    print("  [Scaling] Demonstrating Train/Test Data Leakage safeguards...")
    leakage_demo = demonstrate_scaling_leakage(sample_df, feature='fare_amount', test_size=0.2, random_state=random_state)

    print("  [Scaling] Generating scaling comparison visualization...")
    chart_name = create_scaling_comparison_chart(sample_df, feature_cols=['trip_distance', 'fare_amount'])

    # Assemble summary table comparing before/after for trip_distance and fare_amount
    feature_summary = []
    for col in ['trip_distance', 'fare_amount', 'trip_duration_minutes', 'total_amount']:
        orig = next((s for s in std_res["stats_before"] if s["feature"] == col), None)
        std_s = next((s for s in std_res["stats_after"] if s["feature"] == col), None)
        mm_s = next((s for s in minmax_res["stats_after"] if s["feature"] == col), None)
        rob_s = next((s for s in robust_res["stats_after"] if s["feature"] == col), None)

        if orig:
            feature_summary.append({
                "feature": col,
                "original_mean": orig["mean"],
                "original_std": orig["std"],
                "original_min": orig["min"],
                "original_median": orig["median"],
                "original_max": orig["max"],
                "original_iqr": orig["iqr"],
                # Standard
                "std_mean": std_s["mean"],
                "std_std": std_s["std"],
                "std_min": std_s["min"],
                "std_max": std_s["max"],
                # MinMax
                "minmax_min": mm_s["min"],
                "minmax_max": mm_s["max"],
                "minmax_mean": mm_s["mean"],
                # Robust
                "robust_median": rob_s["median"],
                "robust_iqr": rob_s["iqr"],
                "robust_min": rob_s["min"],
                "robust_max": rob_s["max"],
            })

    return {
        "sample_size": sample_size,
        "train_size": train_size,
        "test_size": test_size,
        "features_scaled": TARGET_SCALING_COLS,
        "chart": chart_name,
        "feature_summary": feature_summary,
        "standard_scaler": {
            "formula": std_res["formula"],
            "description": std_res["description"],
            "outlier_sensitivity": std_res["outlier_sensitivity"],
            "fitted_parameters": std_res["fitted_parameters"],
            "stats_after": std_res["stats_after"]
        },
        "minmax_scaler": {
            "formula": minmax_res["formula"],
            "description": minmax_res["description"],
            "outlier_sensitivity": minmax_res["outlier_sensitivity"],
            "fitted_parameters": minmax_res["fitted_parameters"],
            "stats_after": minmax_res["stats_after"]
        },
        "robust_scaler": {
            "formula": robust_res["formula"],
            "description": robust_res["description"],
            "outlier_sensitivity": robust_res["outlier_sensitivity"],
            "fitted_parameters": robust_res["fitted_parameters"],
            "stats_after": robust_res["stats_after"]
        },
        "leakage_demonstration": leakage_demo,
        "comparison_matrix": get_scaler_comparison_matrix(),
        "model_guidance": get_model_scaling_guidance()
    }
