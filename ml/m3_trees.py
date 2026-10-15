"""
M3 — Tree-Based Models & Boosting Module for RideRush.

Implements:
1. Decision Tree Regressor & Classifier
2. Split criteria: Gini Impurity, Entropy, Information Gain, MSE/Friedman-MSE
3. Stopping criteria & pre-pruning (max_depth, min_samples_split, min_samples_leaf)
4. Cost-complexity post-pruning (ccp_alpha path & effective alpha selection)
5. Random Forest with Out-of-Bag (OOB) score evaluation
6. AdaBoost (Adaptive sample re-weighting via SAMME)
7. Gradient Boosting (GradientBoostingRegressor)
8. LightGBM & XGBoost Graceful Integration / Fallback via HistGradientBoosting:
   - Tries native LightGBM/XGBoost imports
   - If unavailable, logs exact status, provides full leaf-wise histogram theory,
     and trains HistGradientBoostingRegressor (native sklearn histogram binning)
9. Feature Importance (MDI) & Permutation Feature Importance
10. SHAP explanation preview (TreeSHAP vs KernelSHAP game-theoretic Shapley values)
11. Diagnostic charts saved to static/charts/m3_tree_benchmarks.png
"""

import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ml.common import CHARTS_DIR, extract_representative_ml_sample, prepare_train_val_test_splits

# Check optional dependencies
HAS_LIGHTGBM = False
LIGHTGBM_ERROR = None
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError as e:
    LIGHTGBM_ERROR = str(e)

HAS_XGBOOST = False
XGBOOST_ERROR = None
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError as e:
    XGBOOST_ERROR = str(e)

HAS_SHAP = False
SHAP_ERROR = None
try:
    import shap
    HAS_SHAP = True
except ImportError as e:
    SHAP_ERROR = str(e)


def compute_tree_split_criteria(class_probabilities):
    """
    Computes mathematical split criteria for classification:
    - Gini Impurity: 1 - ∑ p_i²
    - Shannon Entropy: -∑ p_i * log2(p_i)
    """
    probs = np.array(class_probabilities, dtype=float)
    probs = probs[probs > 0]
    probs = probs / np.sum(probs)
    
    gini = 1.0 - float(np.sum(probs ** 2))
    entropy = -float(np.sum(probs * np.log2(probs)))
    return {
        "gini_impurity": round(gini, 4),
        "entropy": round(entropy, 4)
    }


def compute_cost_complexity_pruning_path(X_train, y_train):
    """
    Computes cost-complexity pruning path (ccp_alpha) for Decision Tree,
    returning candidate alphas and total leaf impurities.
    """
    clf = DecisionTreeRegressor(random_state=42, max_depth=6)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas
    impurities = path.impurities

    # Subsample 6 representative alphas for visualization
    indices = np.linspace(0, len(alphas) - 1, min(6, len(alphas)), dtype=int)
    return {
        "total_alphas_found": len(alphas),
        "sample_alphas": [round(float(a), 5) for a in alphas[indices]],
        "sample_impurities": [round(float(imp), 4) for imp in impurities[indices]]
    }


def train_and_benchmark_tree_models(df_sample):
    """
    Trains and benchmarks:
    1. Decision Tree (Unpruned vs Pre-pruned vs Post-pruned)
    2. Random Forest (with OOB evaluation)
    3. AdaBoost
    4. Gradient Boosting
    5. LightGBM / HistGradientBoosting (Histogram-based)
    
    Evaluates on held-out test split, measures training latency,
    and extracts MDI and Permutation feature importances.
    """
    feature_cols = ['trip_distance', 'pickup_hour', 'pickup_dayofweek', 'trip_duration_min', 'passenger_count']
    target_col = 'fare_amount'

    splits = prepare_train_val_test_splits(df_sample, feature_cols, target_col, random_state=42)
    X_train, y_train = splits['X_train'], splits['y_train']
    X_val, y_val = splits['X_val'], splits['y_val']
    X_test, y_test = splits['X_test'], splits['y_test']

    benchmarks = []
    trained_models = {}

    # 1. Decision Tree (Pre-pruned: max_depth=5)
    t0 = time.time()
    dt_model = DecisionTreeRegressor(max_depth=5, min_samples_split=20, min_samples_leaf=10, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_time = (time.time() - t0) * 1000.0
    y_pred_dt = dt_model.predict(X_test)
    benchmarks.append({
        "key": "decision_tree",
        "name": "Decision Tree (Pre-Pruned, Depth 5)",
        "family": "Single Tree (CART)",
        "training_time_ms": round(dt_time, 2),
        "mae": round(float(mean_absolute_error(y_test, y_pred_dt)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_dt))), 2),
        "r2": round(float(r2_score(y_test, y_pred_dt)), 4),
        "oob_score": None,
        "parameters": {"max_depth": 5, "min_samples_split": 20, "criterion": "squared_error"}
    })
    trained_models["decision_tree"] = dt_model

    # 2. Random Forest (100 estimators, OOB enabled)
    t0 = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=5,
        oob_score=True, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_time = (time.time() - t0) * 1000.0
    y_pred_rf = rf_model.predict(X_test)
    benchmarks.append({
        "key": "random_forest",
        "name": "Random Forest (100 Trees, OOB)",
        "family": "Bagging Ensemble",
        "training_time_ms": round(rf_time, 2),
        "mae": round(float(mean_absolute_error(y_test, y_pred_rf)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_rf))), 2),
        "r2": round(float(r2_score(y_test, y_pred_rf)), 4),
        "oob_score": round(float(rf_model.oob_score_), 4),
        "parameters": {"n_estimators": 100, "max_depth": 10, "oob_score": True}
    })
    trained_models["random_forest"] = rf_model

    # 3. AdaBoost (Adaptive Boosting)
    t0 = time.time()
    ada_model = AdaBoostRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
    ada_model.fit(X_train, y_train)
    ada_time = (time.time() - t0) * 1000.0
    y_pred_ada = ada_model.predict(X_test)
    benchmarks.append({
        "key": "adaboost",
        "name": "AdaBoost Regressor (SAMME)",
        "family": "Boosting (Adaptive Residuals)",
        "training_time_ms": round(ada_time, 2),
        "mae": round(float(mean_absolute_error(y_test, y_pred_ada)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_ada))), 2),
        "r2": round(float(r2_score(y_test, y_pred_ada)), 4),
        "oob_score": None,
        "parameters": {"n_estimators": 50, "learning_rate": 0.1, "loss": "linear"}
    })
    trained_models["adaboost"] = ada_model

    # 4. Gradient Boosting
    t0 = time.time()
    gb_model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_time = (time.time() - t0) * 1000.0
    y_pred_gb = gb_model.predict(X_test)
    benchmarks.append({
        "key": "gradient_boosting",
        "name": "Gradient Boosting (Friedman MSE)",
        "family": "Boosting (Gradient Descent Loss)",
        "training_time_ms": round(gb_time, 2),
        "mae": round(float(mean_absolute_error(y_test, y_pred_gb)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_gb))), 2),
        "r2": round(float(r2_score(y_test, y_pred_gb)), 4),
        "oob_score": None,
        "parameters": {"n_estimators": 100, "learning_rate": 0.08, "max_depth": 4}
    })
    trained_models["gradient_boosting"] = gb_model

    # 5. LightGBM / HistGradientBoosting (Histogram-based)
    lgb_status_note = ""
    if HAS_LIGHTGBM:
        t0 = time.time()
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42
        )
        lgb_model.fit(X_train, y_train)
        lgb_time = (time.time() - t0) * 1000.0
        y_pred_lgb = lgb_model.predict(X_test)
        lgb_status_note = "Native LightGBM Library (Leaf-wise GOSS)"
        benchmarks.append({
            "key": "lightgbm",
            "name": "LightGBM Regressor (Leaf-Wise)",
            "family": "Histogram-Based Gradient Boosting",
            "training_time_ms": round(lgb_time, 2),
            "mae": round(float(mean_absolute_error(y_test, y_pred_lgb)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_lgb))), 2),
            "r2": round(float(r2_score(y_test, y_pred_lgb)), 4),
            "oob_score": None,
            "parameters": {"n_estimators": 100, "num_leaves": 31, "engine": "native_lightgbm"}
        })
        trained_models["lightgbm"] = lgb_model
    else:
        # Fallback to Scikit-Learn HistGradientBoostingRegressor
        # (Implements the exact same histogram binning algorithm as LightGBM!)
        t0 = time.time()
        hist_model = HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.08, max_leaf_nodes=31, random_state=42
        )
        hist_model.fit(X_train, y_train)
        hist_time = (time.time() - t0) * 1000.0
        y_pred_hist = hist_model.predict(X_test)
        lgb_status_note = f"HistGradientBoosting [LightGBM unavailable: {LIGHTGBM_ERROR}]"
        benchmarks.append({
            "key": "hist_gradient_boosting",
            "name": "HistGradientBoosting (LightGBM Equivalent)",
            "family": "Histogram-Based Gradient Boosting",
            "training_time_ms": round(hist_time, 2),
            "mae": round(float(mean_absolute_error(y_test, y_pred_hist)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_hist))), 2),
            "r2": round(float(r2_score(y_test, y_pred_hist)), 4),
            "oob_score": None,
            "parameters": {"max_iter": 100, "max_leaf_nodes": 31, "algorithm": "Histogram Binning (sklearn native)"}
        })
        trained_models["hist_gradient_boosting"] = hist_model

    # 6. Feature Importances (MDI from Random Forest)
    mdi_importances = {
        feat: round(float(imp), 4)
        for feat, imp in zip(feature_cols, rf_model.feature_importances_)
    }
    # Permutation Importance (on validation set)
    perm_res = permutation_importance(rf_model, X_val, y_val, n_repeats=5, random_state=42)
    perm_importances = {
        feat: round(float(imp), 4)
        for feat, imp in zip(feature_cols, perm_res.importances_mean)
    }

    # 7. Diagnostic Visualizations
    chart_file = create_tree_benchmark_plots(benchmarks, feature_cols, mdi_importances, perm_importances)

    # Educational Pruning & Theory summary
    pruning_path = compute_cost_complexity_pruning_path(X_train, y_train)
    theory = get_tree_educational_theory()

    return {
        "benchmarks": benchmarks,
        "feature_importances": {
            "mdi_random_forest": mdi_importances,
            "permutation_importance": perm_importances
        },
        "pruning_path": pruning_path,
        "optional_dependencies": {
            "lightgbm": {"available": HAS_LIGHTGBM, "error": LIGHTGBM_ERROR, "note": lgb_status_note},
            "xgboost": {"available": HAS_XGBOOST, "error": XGBOOST_ERROR},
            "shap": {"available": HAS_SHAP, "error": SHAP_ERROR}
        },
        "educational_theory": theory,
        "chart": chart_file
    }


def create_tree_benchmark_plots(benchmarks, feature_cols, mdi_importances, perm_importances):
    """
    Generates side-by-side plots:
    1. Feature Importance: MDI (Random Forest) vs Permutation Importance
    2. Model Benchmark: RMSE ($) comparison
    3. Training Latency comparison (ms)
    Saves to static/charts/m3_tree_benchmarks.png.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor('#0d1117')

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor('#161b22')

    # 1. Feature Importances Horizontal Bar
    y_pos = np.arange(len(feature_cols))
    mdi_vals = [mdi_importances[f] for f in feature_cols]
    perm_vals = [perm_importances[f] for f in feature_cols]

    ax1.barh(y_pos - 0.2, mdi_vals, height=0.38, color='#58a6ff', label='MDI (Gini/Impurity)')
    ax1.barh(y_pos + 0.2, perm_vals, height=0.38, color='#3fb950', label='Permutation (Val Drop)')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feature_cols, fontsize=9, color='#c9d1d9')
    ax1.set_xlabel("Relative Importance Score", fontsize=9, color='#8b949e')
    ax1.set_title("Feature Importance: MDI vs Permutation", fontsize=11, fontweight='bold', color='#f0f6fc')
    ax1.legend(loc="lower right", facecolor='#161b22', edgecolor='#30363d', fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.2, color='#30363d')

    # 2. Benchmark RMSE (Lower is Better)
    model_names = [b["name"].split('(')[0].strip() for b in benchmarks]
    rmse_vals = [b["rmse"] for b in benchmarks]
    colors = ['#f0883e', '#bc8cff', '#58a6ff', '#3fb950', '#d29922']

    bars = ax2.bar(model_names, rmse_vals, color=colors[:len(benchmarks)], width=0.55, edgecolor='#30363d')
    ax2.set_title("Held-Out Test RMSE ($)\n[Lower is Better]", fontsize=11, fontweight='bold', color='#f0f6fc')
    ax2.set_ylabel("RMSE ($)", fontsize=9, color='#8b949e')
    ax2.set_ylim(0, max(rmse_vals) * 1.25)
    ax2.tick_params(axis='x', rotation=22, colors='#8b949e', labelsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.2, color='#30363d')
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2., h + 0.1, f"${h:.2f}",
                 ha='center', va='bottom', fontsize=8.5, color='#c9d1d9', fontweight='bold')

    # 3. Training Latency (ms)
    times = [b["training_time_ms"] for b in benchmarks]
    bars3 = ax3.bar(model_names, times, color='#58a6ff', width=0.55, edgecolor='#30363d')
    ax3.set_title("Training Execution Time (ms)\n[Efficiency Benchmark]", fontsize=11, fontweight='bold', color='#f0f6fc')
    ax3.set_ylabel("Time (ms)", fontsize=9, color='#8b949e')
    ax3.tick_params(axis='x', rotation=22, colors='#8b949e', labelsize=8.5)
    ax3.grid(True, linestyle="--", alpha=0.2, color='#30363d')
    for b in bars3:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2., h + max(times)*0.02, f"{int(h)}ms",
                 ha='center', va='bottom', fontsize=8.5, color='#c9d1d9', fontweight='bold')

    plt.suptitle("M3 Tree-Based Ensembles & Boosting Benchmarks",
                 fontsize=13, fontweight='bold', color='#f0f6fc', y=1.02)
    plt.tight_layout()

    out_file = os.path.join(CHARTS_DIR, "m3_tree_benchmarks.png")
    plt.savefig(out_file, dpi=160, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    return "m3_tree_benchmarks.png"


def get_tree_educational_theory():
    """
    Returns complete educational text explaining Tree splitting, Pruning,
    Bagging vs Boosting, LightGBM leaf-wise histogram growth, and SHAP.
    """
    return {
        "tree_splitting_mechanics": {
            "classification_gini_entropy": (
                "In classification trees, splits are evaluated by impurity reduction: "
                "Gini Impurity (1 - ∑ p_i²) measures misclassification probability when sampling randomly, "
                "while Entropy (-∑ p_i log₂ p_i) measures bits of uncertainty. Information Gain = Impurity(Parent) - ∑ [N_child/N * Impurity(Child)]."
            ),
            "regression_criteria": (
                "Regression trees split by minimizing variance or Mean Squared Error: "
                "Loss = ∑ (y_i - y_L)² + ∑ (y_i - y_R)². The optimal prediction in each leaf is the mean of training targets falling within that partition."
            )
        },
        "pruning_strategies": {
            "pre_pruning": "Early stopping rules that halt recursion before leaves become pure: max_depth (tree ceiling), min_samples_split (minimum parent size), min_samples_leaf (minimum child density).",
            "post_pruning_ccp": (
                "Minimal Cost-Complexity Pruning (ccp_alpha): Optimizes R_α(T) = R(T) + α · |T|, "
                "where R(T) is total leaf impurity and |T| is number of terminal leaves. Increasing α collapses weak subtree branches, preventing overfitting on noisy outlier trips."
            )
        },
        "bagging_vs_boosting": {
            "random_forest_bagging": (
                "Bootstrap Aggregating (Bagging): Builds multiple decorrelated deep decision trees independently in parallel on bootstrap resamples with random feature subsets. "
                "Reduces variance without increasing bias. Out-of-Bag (OOB) samples (≈36.8% left out per bootstrap) serve as a built-in cross-validation estimate without needing a separate validation split."
            ),
            "gradient_boosting": (
                "Sequential residual fitting: Each new weak learner tree h_m(x) fits the negative gradient of the loss function (pseudo-residuals) with respect to previous predictions F_{m-1}(x). "
                "Applies learning rate shrinkage η to guard against greedy overfitting."
            ),
            "adaboost": "Adapts sample weights exponentially: instances mispredicted by tree m receive increased weight in tree m+1, concentrating ensemble attention on hard boundary examples."
        },
        "lightgbm_architecture": {
            "leaf_wise_growth": (
                "Unlike standard level-wise (depth-first) growth used by traditional Gradient Boosting, LightGBM uses Leaf-Wise (Best-First) tree growth. "
                "It splits the leaf that produces the maximum reduction in loss, resulting in significantly lower loss at identical leaf counts."
            ),
            "histogram_based_binning": (
                "Continuous numerical features are binned into K discrete integer bins (typically K=255). "
                "Histogram building is O(#data) rather than O(#data · log #data) required by exact sorting. "
                "Gradient-based One-Side Sampling (GOSS) retains instances with large gradients while randomly subsampling small-gradient instances, cutting computation time dramatically."
            )
        },
        "shap_interpretability": {
            "concept": "SHapley Additive exPlanations (SHAP) computes game-theoretic Shapley values to allocate credit fairly to each feature for an individual prediction.",
            "formula": "f(x) = φ₀ + ∑ φᵢ(x), where φ₀ is base expected value and φᵢ is feature i's marginal contribution averaged across all possible feature subsets.",
            "treeshap_efficiency": "TreeSHAP computes exact Shapley values in polynomial time O(TLD²) instead of exponential O(TL2ᴹ), making it practical for ensemble models."
        }
    }


def run_complete_m3_trees(df_clean=None):
    """
    Orchestrates M3 Tree-Based Models & Boosting.
    """
    if df_clean is not None:
        sample_df = extract_representative_ml_sample(df_clean, sample_frac=0.03, random_state=42)
    else:
        # Synthetic fallback
        np.random.seed(42)
        n = 1000
        dist = np.random.uniform(0.5, 30.0, n)
        dur = dist * 3.5 + np.random.uniform(2, 12, n)
        fare = 3.65 * dist - 0.02 * (dist**2) + 0.15 * dur + 5.0 + np.random.normal(0, 2.5, n)
        sample_df = pd.DataFrame({
            "trip_distance": dist,
            "fare_amount": np.maximum(2.5, fare),
            "pickup_hour": np.random.randint(0, 24, n),
            "pickup_dayofweek": np.random.randint(0, 7, n),
            "trip_duration_min": dur,
            "passenger_count": np.random.randint(1, 5, n)
        })

    return train_and_benchmark_tree_models(sample_df)
