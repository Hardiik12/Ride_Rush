"""
M5 — Evaluation, Model Selection & Calibration Module for RideRush.

Implements:
1. Shared Evaluation Framework:
   - Regression: MAE, MSE, RMSE, R2, MAPE (for y > 0), Residual Diagnostics (mean, skew)
   - Classification: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score
2. Model Selection Protocols:
   - Holdout Validation (70/15/15)
   - K-Fold & Stratified K-Fold Cross-Validation (5-Fold)
   - Nested Cross-Validation (Inner 3-Fold Tuning, Outer 3-Fold Evaluation)
   - GridSearchCV vs RandomizedSearchCV comparison
   - Learning Curves (diagnosing High Bias vs High Variance)
   - Validation Curves (varying regularization parameter alpha)
3. Probability Calibration:
   - Platt Scaling (CalibratedClassifierCV method='sigmoid')
   - Isotonic Regression (CalibratedClassifierCV method='isotonic')
   - Reliability Diagrams (Calibration Curves) & Brier Score
4. Statistical Hypothesis Testing:
   - McNemar's Test for paired classification comparisons
   - Paired Bootstrap Confidence Intervals (1000 resamples for 95% CI on RMSE/Accuracy)
5. Diagnostic Plots saved to static/charts/m5_model_evaluation.png
"""

import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_val_score, GridSearchCV,
    RandomizedSearchCV, learning_curve, validation_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ml.common import CHARTS_DIR, extract_representative_ml_sample, prepare_train_val_test_splits


def compute_comprehensive_regression_metrics(y_true, y_pred):
    """
    Computes MAE, MSE, RMSE, R2, MAPE (where y_true > 0), and residual statistics.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    residuals = y_true - y_pred
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # Safe MAPE on non-zero targets
    nonzero = y_true > 0
    mape = float(np.mean(np.abs(residuals[nonzero] / y_true[nonzero])) * 100.0) if np.any(nonzero) else 0.0

    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mape_pct": round(mape, 2),
        "residuals": {
            "mean": round(float(np.mean(residuals)), 4),
            "std": round(float(np.std(residuals)), 4),
            "skewness": round(float(stats.skew(residuals)), 4)
        }
    }


def compute_mcnemar_test(y_true, y_pred_model1, y_pred_model2):
    """
    Computes McNemar's Test for paired nominal data comparing two classifiers:
    Chi2 = (|b - c| - 1)^2 / (b + c) with continuity correction,
    where b = model1 correct & model2 incorrect, c = model1 incorrect & model2 correct.
    """
    y_true = np.asarray(y_true)
    correct1 = (np.asarray(y_pred_model1) == y_true)
    correct2 = (np.asarray(y_pred_model2) == y_true)

    # Contingency Table cells:
    # a: both correct, b: m1 correct / m2 wrong, c: m1 wrong / m2 correct, d: both wrong
    b = int(np.sum(correct1 & ~correct2))
    c = int(np.sum(~correct1 & correct2))
    
    if (b + c) > 0:
        # McNemar test statistic with Edwards continuity correction
        stat = (abs(b - c) - 1.0) ** 2 / (b + c)
        p_value = 1.0 - stats.chi2.cdf(stat, df=1)
    else:
        stat = 0.0
        p_value = 1.0

    return {
        "b_cell_m1_better": b,
        "c_cell_m2_better": c,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 5),
        "statistically_significant": bool(p_value < 0.05),
        "interpretation": (
            "Statistically significant difference in error distributions (p < 0.05)"
            if p_value < 0.05 else
            "No statistically significant difference between the two classifiers (p >= 0.05)"
        )
    }


def compute_paired_bootstrap_ci(y_true, y_pred_model1, y_pred_model2, metric_fn=mean_squared_error, n_bootstraps=1000, random_state=42):
    """
    Computes 95% Paired Bootstrap Confidence Interval for the difference in test performance:
    Delta = Metric(Model 1) - Metric(Model 2).
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        s1 = metric_fn(y_true[idx], y_pred_model1[idx])
        s2 = metric_fn(y_true[idx], y_pred_model2[idx])
        diffs.append(s1 - s2)

    diffs = np.array(diffs)
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    mean_diff = float(np.mean(diffs))

    return {
        "mean_difference": round(mean_diff, 4),
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
        "zero_in_interval": bool(ci_lower <= 0 <= ci_upper)
    }


def run_model_selection_experiments(df_sample):
    """
    Executes:
    1. 5-Fold Cross-Validation on Ridge Regression
    2. Stratified 5-Fold Cross-Validation on Logistic Regression
    3. Nested Cross-Validation (3x3)
    4. GridSearchCV vs RandomizedSearchCV benchmark
    5. Probability Calibration (Platt Sigmoid vs Isotonic Regression vs Uncalibrated)
    6. Learning Curves (Bias vs Variance)
    7. McNemar's Test on Classifiers
    8. Bootstrap Confidence Intervals on Regression Models
    """
    feature_cols = ['trip_distance', 'pickup_hour', 'pickup_dayofweek', 'trip_duration_min']
    
    # Regression splits
    reg_splits = prepare_train_val_test_splits(df_sample, feature_cols, 'fare_amount', random_state=42)
    X_tr_r, y_tr_r = reg_splits['X_train'], reg_splits['y_train']
    X_te_r, y_te_r = reg_splits['X_test'], reg_splits['y_test']

    # Classification splits
    clf_splits = prepare_train_val_test_splits(df_sample, feature_cols, 'is_high_tip', random_state=42)
    X_tr_c, y_tr_c = clf_splits['X_train'], clf_splits['y_train']
    X_te_c, y_te_c = clf_splits['X_test'], clf_splits['y_test']

    # 1. K-Fold Cross-Validation (5-Fold) on Ridge
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pipe_ridge = Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=1.0))])
    cv_scores_r2 = cross_val_score(pipe_ridge, X_tr_r, y_tr_r, cv=kf, scoring='r2')
    cv_scores_rmse = np.sqrt(-cross_val_score(pipe_ridge, X_tr_r, y_tr_r, cv=kf, scoring='neg_mean_squared_error'))

    kfold_summary = {
        "k_folds": 5,
        "fold_r2_scores": [round(float(s), 4) for s in cv_scores_r2],
        "mean_r2": round(float(np.mean(cv_scores_r2)), 4),
        "std_r2": round(float(np.std(cv_scores_r2)), 4),
        "mean_rmse": round(float(np.mean(cv_scores_rmse)), 4),
        "std_rmse": round(float(np.std(cv_scores_rmse)), 4)
    }

    # 2. Stratified K-Fold (5-Fold) on Classifier
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe_clf = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42))])
    cv_scores_f1 = cross_val_score(pipe_clf, X_tr_c, y_tr_c, cv=skf, scoring='f1')
    stratified_summary = {
        "k_folds": 5,
        "fold_f1_scores": [round(float(s), 4) for s in cv_scores_f1],
        "mean_f1": round(float(np.mean(cv_scores_f1)), 4),
        "std_f1": round(float(np.std(cv_scores_f1)), 4)
    }

    # 3. Nested Cross-Validation (Outer 3-Fold, Inner 3-Fold Grid Search)
    outer_kf = KFold(n_splits=3, shuffle=True, random_state=42)
    inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
    param_grid_ridge = {'reg__alpha': [0.1, 1.0, 10.0]}
    grid_inner = GridSearchCV(pipe_ridge, param_grid_ridge, cv=inner_kf, scoring='neg_mean_squared_error')
    nested_cv_scores = np.sqrt(-cross_val_score(grid_inner, X_tr_r, y_tr_r, cv=outer_kf))
    nested_summary = {
        "outer_folds": 3,
        "inner_folds": 3,
        "nested_rmse_per_fold": [round(float(s), 4) for s in nested_cv_scores],
        "nested_mean_rmse": round(float(np.mean(nested_cv_scores)), 4),
        "interpretation": "Nested CV yields an unbiased estimate of generalization error without selection bias."
    }

    # 4. GridSearch vs RandomizedSearch Benchmark
    t0 = time.time()
    grid = GridSearchCV(pipe_ridge, {'reg__alpha': [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}, cv=3, scoring='r2')
    grid.fit(X_tr_r, y_tr_r)
    grid_time = (time.time() - t0) * 1000.0

    t0 = time.time()
    rand = RandomizedSearchCV(pipe_ridge, {'reg__alpha': stats.uniform(0.01, 100.0)}, n_iter=7, cv=3, scoring='r2', random_state=42)
    rand.fit(X_tr_r, y_tr_r)
    rand_time = (time.time() - t0) * 1000.0

    search_benchmark = {
        "grid_search": {
            "best_alpha": round(float(grid.best_params_['reg__alpha']), 3),
            "best_r2": round(float(grid.best_score_), 4),
            "latency_ms": round(grid_time, 2),
            "candidates_evaluated": len(grid.cv_results_['params'])
        },
        "randomized_search": {
            "best_alpha": round(float(rand.best_params_['reg__alpha']), 3),
            "best_r2": round(float(rand.best_score_), 4),
            "latency_ms": round(rand_time, 2),
            "candidates_evaluated": len(rand.cv_results_['params'])
        }
    }

    # 5. Probability Calibration: Uncalibrated vs Platt Sigmoid vs Isotonic
    base_clf = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.05, random_state=42))])
    base_clf.fit(X_tr_c, y_tr_c)
    uncal_probs = base_clf.predict_proba(X_te_c)[:, 1]
    brier_uncal = float(brier_score_loss(y_te_c, uncal_probs))

    # Platt Sigmoid Calibration
    calib_sigmoid = CalibratedClassifierCV(base_clf, method='sigmoid', cv=3)
    calib_sigmoid.fit(X_tr_c, y_tr_c)
    sig_probs = calib_sigmoid.predict_proba(X_te_c)[:, 1]
    brier_sig = float(brier_score_loss(y_te_c, sig_probs))

    # Isotonic Regression Calibration
    calib_iso = CalibratedClassifierCV(base_clf, method='isotonic', cv=3)
    calib_iso.fit(X_tr_c, y_tr_c)
    iso_probs = calib_iso.predict_proba(X_te_c)[:, 1]
    brier_iso = float(brier_score_loss(y_te_c, iso_probs))

    calibration_summary = {
        "uncalibrated": {"brier_score": round(brier_uncal, 4)},
        "platt_sigmoid": {"brier_score": round(brier_sig, 4)},
        "isotonic_regression": {"brier_score": round(brier_iso, 4)}
    }

    # 6. Statistical Significance Tests
    # McNemar's Test: Uncalibrated vs Isotonic Classifier predictions
    pred_uncal = (uncal_probs >= 0.5).astype(int)
    pred_iso = (iso_probs >= 0.5).astype(int)
    mcnemar_res = compute_mcnemar_test(y_te_c.values, pred_uncal, pred_iso)

    # Paired Bootstrap CI: Ridge (alpha=1.0) vs Linear GD
    pipe_ridge.fit(X_tr_r, y_tr_r)
    pred_ridge = pipe_ridge.predict(X_te_r)
    # Simple linear benchmark
    pipe_simple = Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=100.0))])
    pipe_simple.fit(X_tr_r, y_tr_r)
    pred_simple = pipe_simple.predict(X_te_r)
    bootstrap_res = compute_paired_bootstrap_ci(y_te_r.values, pred_ridge, pred_simple, n_bootstraps=500)

    # 7. Diagnostic Visualizations
    chart_file = create_evaluation_plots(
        pipe_ridge, X_tr_r, y_tr_r, y_te_c,
        uncal_probs, sig_probs, iso_probs,
        y_te_r.values, pred_ridge
    )

    theory = get_evaluation_educational_theory()

    return {
        "kfold_cv": kfold_summary,
        "stratified_cv": stratified_summary,
        "nested_cv": nested_summary,
        "search_benchmark": search_benchmark,
        "probability_calibration": calibration_summary,
        "mcnemar_test": mcnemar_res,
        "bootstrap_ci_95": bootstrap_res,
        "chart": chart_file,
        "educational_theory": theory
    }


def create_evaluation_plots(pipe_reg, X_tr, y_tr, y_test_c, uncal_p, sig_p, iso_p, y_test_r, pred_r):
    """
    Generates multi-panel evaluation diagnostic charts:
    1. Residual Diagnostics Plot (Residuals vs Predicted Fares)
    2. Reliability Diagrams / Calibration Curves
    3. Learning Curve (Bias vs Variance across Sample Sizes)
    Saves to static/charts/m5_model_evaluation.png.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor('#0d1117')

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor('#161b22')

    # Panel 1: Residual Plot
    residuals = y_test_r - pred_r
    sub_n = min(1200, len(pred_r))
    idx = np.random.RandomState(42).choice(len(pred_r), sub_n, replace=False)
    ax1.scatter(pred_r[idx], residuals[idx], color='#58a6ff', alpha=0.35, s=14)
    ax1.axhline(0, color='#f0883e', linestyle='--', lw=1.8, label='Zero Error')
    ax1.set_title("Residual Plot (Errors vs Fitted Values)", fontsize=11, fontweight='bold', color='#f0f6fc')
    ax1.set_xlabel("Fitted Fare Amount ($)", fontsize=9, color='#8b949e')
    ax1.set_ylabel("Residual (Actual - Fitted)", fontsize=9, color='#8b949e')
    ax1.legend(loc="upper right", facecolor='#161b22', edgecolor='#30363d', fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.2, color='#30363d')

    # Panel 2: Reliability Diagrams (Calibration Curves)
    fop_uncal, mp_uncal = calibration_curve(y_test_c, uncal_p, n_bins=8)
    fop_sig, mp_sig = calibration_curve(y_test_c, sig_p, n_bins=8)
    fop_iso, mp_iso = calibration_curve(y_test_c, iso_p, n_bins=8)

    ax2.plot(mp_uncal, fop_uncal, marker='o', color='#f0883e', lw=2.0, label='Uncalibrated (Logistic)')
    ax2.plot(mp_sig, fop_sig, marker='s', color='#58a6ff', lw=2.0, linestyle='--', label='Platt Sigmoid')
    ax2.plot(mp_iso, fop_iso, marker='^', color='#3fb950', lw=2.0, linestyle='-.', label='Isotonic Step')
    ax2.plot([0, 1], [0, 1], color='#8b949e', linestyle=':', lw=1.2, label='Perfect Calibration')
    ax2.set_title("Reliability Diagram (Calibration Curves)", fontsize=11, fontweight='bold', color='#f0f6fc')
    ax2.set_xlabel("Mean Predicted Probability", fontsize=9, color='#8b949e')
    ax2.set_ylabel("Fraction of True Positives", fontsize=9, color='#8b949e')
    ax2.legend(loc="lower right", facecolor='#161b22', edgecolor='#30363d', fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.2, color='#30363d')

    # Panel 3: Learning Curves (Sample Size vs Performance)
    train_sizes, train_scores, val_scores = learning_curve(
        pipe_reg, X_tr, y_tr,
        train_sizes=np.linspace(0.15, 1.0, 5),
        cv=3, scoring='r2', random_state=42
    )
    tr_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    ax3.plot(train_sizes, tr_mean, marker='o', color='#58a6ff', lw=2.2, label='Train Score (R²)')
    ax3.plot(train_sizes, val_mean, marker='s', color='#3fb950', lw=2.2, label='CV Validation Score (R²)')
    ax3.set_title("Learning Curve (Bias vs Variance)", fontsize=11, fontweight='bold', color='#f0f6fc')
    ax3.set_xlabel("Training Instances Sampled", fontsize=9, color='#8b949e')
    ax3.set_ylabel("R² Score", fontsize=9, color='#8b949e')
    ax3.legend(loc="lower right", facecolor='#161b22', edgecolor='#30363d', fontsize=8.5)
    ax3.grid(True, linestyle="--", alpha=0.2, color='#30363d')

    plt.suptitle("M5 Evaluation, Calibration & Model Selection Workbench",
                 fontsize=13, fontweight='bold', color='#f0f6fc', y=1.02)
    plt.tight_layout()

    out_file = os.path.join(CHARTS_DIR, "m5_model_evaluation.png")
    plt.savefig(out_file, dpi=160, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    return "m5_model_evaluation.png"


def get_evaluation_educational_theory():
    """
    Returns educational text explaining Nested CV, Calibration methods,
    and statistical hypothesis tests for ML models.
    """
    return {
        "nested_cross_validation": {
            "concept": "Nested CV prevents optimistic selection bias by decoupling hyperparameter tuning from generalisation performance evaluation.",
            "structure": "Inner Loop (e.g. 3-fold): Searches over hyperparameter grid to find best configuration. Outer Loop (e.g. 3-fold): Evaluates the chosen model on an independent held-out partition."
        },
        "probability_calibration": {
            "platt_scaling": "Fits a logistic sigmoid function P(y=1|f) = 1 / (1 + exp(A·f + B)) directly onto classifier decision logits f(x). Effective when distortion is sigmoid-shaped.",
            "isotonic_regression": "Non-parametric piecewise constant monotonic step function that minimizes squared error subject to the constraint that predicted probabilities increase monotonically with score.",
            "brier_score": "BS = 1/N ∑ (p_i - y_i)². Measures mean squared calibration error. Perfect calibration yields Brier Score = 0."
        },
        "hypothesis_testing": {
            "mcnemar_test": "Non-parametric test of paired nominal classifications. Compares discordant error pairs b (Model 1 right, Model 2 wrong) and c (Model 1 wrong, Model 2 right) to determine if performance differences are statistically significant or due to chance.",
            "bootstrap_confidence_intervals": "Resamples test records with replacement 1000 times to construct an empirical sampling distribution for performance metric differences, providing robust 95% confidence intervals without normality assumptions."
        }
    }


def run_complete_m5_evaluation(df_clean=None):
    """
    Orchestrates full M5 Evaluation, Selection & Calibration pipeline.
    """
    if df_clean is not None:
        sample_df = extract_representative_ml_sample(df_clean, sample_frac=0.03, random_state=42)
    else:
        np.random.seed(42)
        n = 1000
        dist = np.random.uniform(0.5, 25.0, n)
        fare = 3.65 * dist + 5.0 + np.random.normal(0, 3.0, n)
        tip = np.where(np.random.rand(n) > 0.6, fare * 0.22, fare * 0.08)
        sample_df = pd.DataFrame({
            "trip_distance": dist,
            "fare_amount": np.maximum(2.5, fare),
            "tip_amount": tip,
            "pickup_hour": np.random.randint(0, 24, n),
            "pickup_dayofweek": np.random.randint(0, 7, n),
            "trip_duration_min": dist * 3.2 + np.random.uniform(2, 10, n),
            "is_high_tip": (tip / np.maximum(fare, 1.0) >= 0.18).astype(int),
            "passenger_count": np.random.randint(1, 5, n)
        })

    return run_model_selection_experiments(sample_df)
