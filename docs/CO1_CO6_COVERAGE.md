# RideRush — Complete CO1 to CO6 Course Outcomes Traceability Matrix

This document provides the definitive Course Outcome (CO1–CO6) Traceability Matrix for the **RideRush NYC Taxi Big Data Machine Learning Platform**. It maps every curriculum topic, mathematical formulation, and operational component to its exact implementation file, test evidence, user interface/API route, and verified status.

---

## Executive Summary: Course Outcome Status Overview

| Course Outcome | Focus Area | Total Topics | Verification Status | Implementation Modules | Key Route / Interface |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **CO1** | Big Data Wrangling, Cleaning, Outlier Analysis, Imputation, Scaling & Duplicates | 20 Topics | **COMPLETE** | `ml/data_loader.py`, `ml/data_cleaning.py`, `ml/outliers.py`, `ml/missing_values.py`, `ml/scaling.py`, `ml/duplicates.py`, `ml/eda.py` | `/data`, `/cleaning`, `/eda` |
| **CO2** | Supervised Learning: Regression Models, Gradient Descent & Regularization | 8 Topics | **COMPLETE** | `ml/linear_regression.py`, `ml/regression_workbench.py` | `/prediction`, `/workbench?tab=regression`, `/api/workbench` |
| **CO3** | Supervised Learning: Classification, Imbalance & Target Transformations | 7 Topics | **COMPLETE** | `ml/m2_classification.py` | `/workbench?tab=classification` |
| **CO4** | Tree-Based Models, Pruning, Ensemble Boosting & Explainability | 11 Topics | **COMPLETE** | `ml/m3_trees.py` | `/workbench?tab=trees` |
| **CO5** | Unsupervised Learning, Clustering, PCA & Anomaly Detection | 10 Topics | **COMPLETE** | `ml/m4_unsupervised.py` | `/workbench?tab=unsupervised` |
| **CO6** | Rigorous Evaluation Protocols, Calibration & Production MLOps Engineering | 20 Topics | **COMPLETE** | `ml/m5_evaluation.py`, `ml/m6_engineering.py`, `ml/m1_lifecycle.py`, `ml/common.py` | `/workbench?tab=evaluation`, `/workbench?tab=engineering`, `/workbench?tab=lifecycle`, `/api/workbench/*` |
| **TOTAL** | **Full Machine Learning Lifecycle & Engineering System** | **76 Topics** | **100% VERIFIED** | **15 Core Modules** | **8 Interactive Dashboard Tabs & 7 REST APIs** |

---

## Detailed Traceability Matrix

### Course Outcome 1 (CO1): Big Data Acquisition, Data Cleaning, Outlier Analysis, Imputation, Scaling & Duplicates

> **CO1 Objective**: Understand big data characteristics, inspect data quality, apply out-of-core chunk filtering via Dask, perform statistical outlier detection and treatment, handle missing values across MCAR/MAR/MNAR classifications with leakage prevention, evaluate feature scaling algorithms and duplicate record operational risks, and generate comprehensive exploratory visualizations.

| Topic ID | Required Topic | Mathematical / Algorithmic Formulation | Implementation File & Function | Test Evidence | Dashboard / API Route | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CO1.1** | Big Data Lazy Loading | Out-of-core chunked partition ingestion | [`ml/data_loader.py`](file:///Users/hardik/RideRush/ml/data_loader.py): `load_taxi_data()` | `test_dask_loader` | `/data` | **COMPLETE** |
| **CO1.2** | Data Understanding & Metadata | Row count, column schema, memory footprints | [`ml/data_loader.py`](file:///Users/hardik/RideRush/ml/data_loader.py): `get_dataset_info()` | `test_dataset_exists` | `/data` | **COMPLETE** |
| **CO1.3** | Data Quality Analysis | Null counts, negative fares, duration violations | [`ml/data_cleaning.py`](file:///Users/hardik/RideRush/ml/data_cleaning.py): `check_data_quality()` | `test_cleaning_route` | `/cleaning` | **COMPLETE** |
| **CO1.4** | Dask Logical Filtering | Vectorized predicate masks (passengers, bounds) | [`ml/data_cleaning.py`](file:///Users/hardik/RideRush/ml/data_cleaning.py): `clean_taxi_data()` | `test_cleaning_route` | `/cleaning` | **COMPLETE** |
| **CO1.5** | IQR Outlier Identification | $Q_1, Q_2, Q_3, \text{IQR} = Q_3 - Q_1$ | [`ml/outliers.py`](file:///Users/hardik/RideRush/ml/outliers.py): `calculate_iqr_statistics()` | `test_co1_outliers_module` | `/eda#outliers` | **COMPLETE** |
| **CO1.6** | Statistical Tukey Bounds | $[\text{Lower} = Q_1 - 1.5\cdot\text{IQR},\, \text{Upper} = Q_3 + 1.5\cdot\text{IQR}]$ | [`ml/outliers.py`](file:///Users/hardik/RideRush/ml/outliers.py): `calculate_iqr_statistics()` | `test_co1_outliers_module` | `/eda#outliers` | **COMPLETE** |
| **CO1.7** | Outlier Treatment Strategies | Removal vs. Retention vs. Winsorization ($5\%\text{--}95\%$) | [`ml/outliers.py`](file:///Users/hardik/RideRush/ml/outliers.py): `evaluate_outlier_treatments()` | `test_co1_outliers_module` | `/eda#outliers` | **COMPLETE** |
| **CO1.8** | Outlier Boxplot Diagnostics | Five-number summary visualization | [`ml/outliers.py`](file:///Users/hardik/RideRush/ml/outliers.py): `generate_outlier_boxplots()` | `test_eda_co1_rendering` | `static/charts/outlier_boxplots.png` | **COMPLETE** |
| **CO1.9** | Missing Value Audit | Column-wise null incidence and percentages | [`ml/missing_values.py`](file:///Users/hardik/RideRush/ml/missing_values.py): `audit_missing_values()` | `test_co1_missing_values_module` | `/eda#missing-values` | **COMPLETE** |
| **CO1.10** | Missingness Mechanisms | MCAR vs. MAR vs. MNAR theoretical taxonomy | [`ml/missing_values.py`](file:///Users/hardik/RideRush/ml/missing_values.py): `classify_missingness_mechanism()` | `test_co1_missing_values_module` | `/eda#missing-values` | **COMPLETE** |
| **CO1.11** | Deletion Techniques | Complete case row deletion vs. feature column deletion | [`ml/missing_values.py`](file:///Users/hardik/RideRush/ml/missing_values.py): `delete_missing_rows()`, `delete_missing_columns()` | `test_co1_missing_values_module` | `/eda#missing-values` | **COMPLETE** |
| **CO1.12** | Mean & Median Imputation | Unconditional central tendency replacement | [`ml/missing_values.py`](file:///Users/hardik/RideRush/ml/missing_values.py): `mean_imputation()`, `median_imputation()` | `test_co1_missing_values_module` | `/eda#missing-values` | **COMPLETE** |
| **CO1.13** | Advanced Imputation | KNN ($K=5$) and MICE (`IterativeImputer`) | [`ml/missing_values.py`](file:///Users/hardik/RideRush/ml/missing_values.py): `knn_imputation()`, `iterative_imputation()` | `test_co1_missing_values_module` | `/eda#missing-values` | **COMPLETE** |
| **CO1.14** | Missingness Indicators | Binary dummy flag $I_m \in \{0, 1\}$ | [`ml/missing_values.py`](file:///Users/hardik/RideRush/ml/missing_values.py): `create_missing_indicators()` | `test_co1_missing_values_module` | `/eda#missing-values` | **COMPLETE** |
| **CO1.15** | Imputation Leakage Prevention | Strict training-only imputer fitting: $\text{fit}(X_{\text{train}})$ | [`ml/missing_values.py`](file:///Users/hardik/RideRush/ml/missing_values.py): `fit_imputer_on_train_only()` | `test_co1_missing_values_module` | `/eda#missing-values` | **COMPLETE** |
| **CO1.16** | Z-Score Standardization | $z = \frac{x - \mu}{\sigma}$, unit variance, zero mean | [`ml/scaling.py`](file:///Users/hardik/RideRush/ml/scaling.py): `standard_scaling()` | `test_co1_scaling_module` | `/eda#scaling` | **COMPLETE** |
| **CO1.17** | Min-Max Scaling | $x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \in [0, 1]$ | [`ml/scaling.py`](file:///Users/hardik/RideRush/ml/scaling.py): `minmax_scaling()` | `test_co1_scaling_module` | `/eda#scaling` | **COMPLETE** |
| **CO1.18** | Robust Scaling (IQR) | $x_{\text{robust}} = \frac{x - \text{Median}}{\text{IQR}}$ | [`ml/scaling.py`](file:///Users/hardik/RideRush/ml/scaling.py): `robust_scaling()` | `test_co1_scaling_module` | `/eda#scaling` | **COMPLETE** |
| **CO1.19** | Scaler Leakage Safeguards | Measuring parameter shift: $\Delta \mu = \|\mu_{\text{comb}} - \mu_{\text{tr}}\|$ | [`ml/scaling.py`](file:///Users/hardik/RideRush/ml/scaling.py): `demonstrate_scaling_leakage()` | `test_co1_scaling_module` | `/eda#scaling` | **COMPLETE** |
| **CO1.20** | Duplicate Records Handling | Exact vs. Subset duplicates; legitimate trip risks | [`ml/duplicates.py`](file:///Users/hardik/RideRush/ml/duplicates.py): `audit_exact_duplicates()`, `audit_subset_duplicates()` | `test_co1_duplicates_module` | `/eda#duplicates` | **COMPLETE** |

---

### Course Outcome 2 (CO2): Supervised Learning — Regression Foundations & Regularization

> **CO2 Objective**: Formulate supervised regression tasks, derive the Mean Squared Error cost function, implement custom Batch Gradient Descent with cost convergence tracking, derive and execute the closed-form Normal Equation solution, evaluate non-linear Polynomial regression expansions, and benchmark Ridge, Lasso, and Elastic Net regularization under strict train/test data leakage safeguards.

| Topic ID | Required Topic | Mathematical / Algorithmic Formulation | Implementation File & Function | Test Evidence | Dashboard / API Route | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CO2.1** | Simple Linear Regression | Hypothesis: $\hat{y} = w \cdot x + b$ | [`ml/linear_regression.py`](file:///Users/hardik/RideRush/ml/linear_regression.py): `SimpleLinearRegressionGD` | `test_linear_regression_gd_model` | `/prediction` | **COMPLETE** |
| **CO2.2** | Batch Gradient Descent | $w := w - \alpha \frac{\partial J}{\partial w},\, b := b - \alpha \frac{\partial J}{\partial b}$ | [`ml/linear_regression.py`](file:///Users/hardik/RideRush/ml/linear_regression.py): `SimpleLinearRegressionGD.fit()` | `test_linear_regression_gd_model` | `static/charts/cost_convergence.png` | **COMPLETE** |
| **CO2.3** | Normal Equation Closed-Form OLS | $\theta = (X^T X)^{-1} X^T y$ | [`ml/regression_workbench.py`](file:///Users/hardik/RideRush/ml/regression_workbench.py): `solve_normal_equation()` | `test_m2_normal_equation_analytical_solution` | `/workbench?tab=regression` | **COMPLETE** |
| **CO2.4** | Polynomial Feature Expansion | Degree 2: $\phi(x) = [1, x, x^2]$, Runge's bounds | [`ml/regression_workbench.py`](file:///Users/hardik/RideRush/ml/regression_workbench.py): `train_workbench_models()` | `test_workbench_polynomial_feature_generation` | `/workbench?tab=regression` | **COMPLETE** |
| **CO2.5** | Ridge Regression ($L_2$ Penalty) | $J(\theta) = \text{MSE} + \frac{\lambda}{2} \|\theta\|_2^2$ | [`ml/regression_workbench.py`](file:///Users/hardik/RideRush/ml/regression_workbench.py): `train_workbench_models()` | `test_workbench_regularization_parameters_configurable` | `/workbench?tab=regression` | **COMPLETE** |
| **CO2.6** | Lasso Regression ($L_1$ Penalty) | $J(\theta) = \text{MSE} + \lambda \|\theta\|_1$ (Sparsity) | [`ml/regression_workbench.py`](file:///Users/hardik/RideRush/ml/regression_workbench.py): `train_workbench_models()` | `test_workbench_regularization_parameters_configurable` | `/workbench?tab=regression` | **COMPLETE** |
| **CO2.7** | Elastic Net ($L_1 + L_2$ Combined) | $J(\theta) = \text{MSE} + r\lambda\|\theta\|_1 + \frac{1-r}{2}\lambda\|\theta\|_2^2$ | [`ml/regression_workbench.py`](file:///Users/hardik/RideRush/ml/regression_workbench.py): `train_workbench_models()` | `test_workbench_regularization_parameters_configurable` | `/workbench?tab=regression` | **COMPLETE** |
| **CO2.8** | Leakage-Free Preprocessing | Scikit-learn Pipeline fitting strictly on train | [`ml/regression_workbench.py`](file:///Users/hardik/RideRush/ml/regression_workbench.py): `train_workbench_models()` | `test_workbench_training_only_leakage_prevention` | `/workbench?tab=regression` | **COMPLETE** |

---

### Course Outcome 3 (CO3): Supervised Learning — Classification, Imbalance & Target Transformations

> **CO3 Objective**: Formulate binary and multi-class classification problems, implement and verify Sigmoid and Softmax activations, calculate log-loss Cross-Entropy, mitigate severe real-world class imbalance, apply out-of-fold target encoding without label leakage, and evaluate classification diagnostics (ROC-AUC, PR-AUC).

| Topic ID | Required Topic | Mathematical / Algorithmic Formulation | Implementation File & Function | Test Evidence | Dashboard / API Route | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CO3.1** | Binary Logistic Regression | Predict high tip: $\text{tip\_fraction} \ge 18\%$ | [`ml/m2_classification.py`](file:///Users/hardik/RideRush/ml/m2_classification.py): `train_and_evaluate_classification_models()` | `test_m2_classification_models_and_metrics` | `/workbench?tab=classification` | **COMPLETE** |
| **CO3.2** | Multinomial Logistic Regression | Predict fare tiers: Low ($< \$12$), Medium, High | [`ml/m2_classification.py`](file:///Users/hardik/RideRush/ml/m2_classification.py): `train_and_evaluate_classification_models()` | `test_m2_classification_models_and_metrics` | `/workbench?tab=classification` | **COMPLETE** |
| **CO3.3** | Sigmoid Activation | $\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$ | [`ml/m2_classification.py`](file:///Users/hardik/RideRush/ml/m2_classification.py): `sigmoid()` | `test_m2_classification_models_and_metrics` | `/workbench?tab=classification` | **COMPLETE** |
| **CO3.4** | Softmax Activation | $\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}},\, \sum P = 1$ | [`ml/m2_classification.py`](file:///Users/hardik/RideRush/ml/m2_classification.py): `softmax()` | `test_m2_classification_models_and_metrics` | `/workbench?tab=classification` | **COMPLETE** |
| **CO3.5** | Cross-Entropy Loss | $L = -\frac{1}{N} \sum [y \ln \hat{y} + (1-y)\ln(1-\hat{y})]$ | [`ml/m2_classification.py`](file:///Users/hardik/RideRush/ml/m2_classification.py): `compute_cross_entropy_loss()` | `test_m2_classification_models_and_metrics` | `/workbench?tab=classification` | **COMPLETE** |
| **CO3.6** | Class Imbalance Mitigation | Class re-weighting: $w_c = \frac{N}{K \cdot N_c}$ | [`ml/m2_classification.py`](file:///Users/hardik/RideRush/ml/m2_classification.py): `train_and_evaluate_classification_models()` | `test_m2_classification_models_and_metrics` | `/workbench?tab=classification` | **COMPLETE** |
| **CO3.7** | Out-of-Fold (OOF) Target Encoding | $K$-Fold cross-validated prior smoothing | [`ml/m2_classification.py`](file:///Users/hardik/RideRush/ml/m2_classification.py): `out_of_fold_target_encoding()` | `test_m2_target_encoding_leakage_safety` | `/workbench?tab=classification` | **COMPLETE** |

---

### Course Outcome 4 (CO4): Tree-Based Models, Ensemble Methods & Explainability

> **CO4 Objective**: Implement CART decision tree models, compute Gini impurity, Shannon entropy, and variance reduction split criteria, formulate pre-pruning stopping criteria and cost-complexity post-pruning paths, benchmark Random Forest ensembles with Out-Of-Bag evaluation, implement AdaBoost, Gradient Boosting, and histogram-based boosting (LightGBM/HistGradientBoosting), contrast MDI vs. Permutation feature importances, and present game-theoretic SHAP explainability.

| Topic ID | Required Topic | Mathematical / Algorithmic Formulation | Implementation File & Function | Test Evidence | Dashboard / API Route | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CO4.1** | CART Decision Tree Regressor | Recursive binary partitioning minimizing MSE | [`ml/m3_trees.py`](file:///Users/hardik/RideRush/ml/m3_trees.py): `train_and_benchmark_tree_models()` | `test_m3_tree_models_and_ensembles` | `/workbench?tab=trees` | **COMPLETE** |
| **CO4.2** | Gini Impurity & Shannon Entropy | $I_G = 1 - \sum p_i^2,\quad H = -\sum p_i \log_2 p_i$ | [`ml/m3_trees.py`](file:///Users/hardik/RideRush/ml/m3_trees.py): `compute_tree_split_criteria()` | `test_m3_pruning_path_and_criteria` | `/workbench?tab=trees` | **COMPLETE** |
| **CO4.3** | Pre-Pruning Early Stopping | `max_depth=5`, `min_samples_split=20`, `min_samples_leaf=10` | [`ml/m3_trees.py`](file:///Users/hardik/RideRush/ml/m3_trees.py): `train_and_benchmark_tree_models()` | `test_m3_tree_models_and_ensembles` | `/workbench?tab=trees` | **COMPLETE** |
| **CO4.4** | Cost-Complexity Post-Pruning (CCP) | $R_\alpha(T) = R(T) + \alpha |T|$, effective $\alpha$ sequence | [`ml/m3_trees.py`](file:///Users/hardik/RideRush/ml/m3_trees.py): `compute_cost_complexity_pruning_path()` | `test_m3_pruning_path_and_criteria` | `/workbench?tab=trees` | **COMPLETE** |
| **CO4.5** | Random Forest with OOB Evaluation | Bootstrap aggregation with built-in validation ($R^2_{\text{OOB}}$) | [`ml/m3_trees.py`](file:///Users/hardik/RideRush/ml/m3_trees.py): `train_and_benchmark_tree_models()` | `test_m3_tree_models_and_ensembles` | `/workbench?tab=trees` | **COMPLETE** |
| **CO4.6** | AdaBoost Regressor | SAMME.R adaptive exponential error re-weighting | [`ml/m3_trees.py`](file:///Users/hardik/RideRush/ml/m3_trees.py): `train_and_benchmark_tree_models()` | `test_m3_tree_models_and_ensembles` | `/workbench?tab=trees` | **COMPLETE** |
| **CO4.7** | Gradient Boosting Regressor | Forward stagewise negative gradient residual fitting | [`ml/m3_trees.py`](file:///Users/hardik/RideRush/ml/m3_trees.py): `train_and_benchmark_tree_models()` | `test_m3_tree_models_and_ensembles` | `/workbench?tab=trees` | **COMPLETE** |
| **CO4.8** | HistGradientBoosting (LightGBM) | Discrete integer binning ($K=255$), $\mathcal{O}(\#\text{data})$ | [`ml/m3_trees.py`](file:///Users/hardik/RideRush/ml/m3_trees.py): `train_and_benchmark_tree_models()` | `test_m3_tree_models_and_ensembles` | `/workbench?tab=trees` | **COMPLETE** |
| **CO4.9** | Mean Decrease Impurity (MDI) | Summed variance drop across splits | [`ml/m3_trees.py`](file:///Users/hardik/RideRush/ml/m3_trees.py): `train_and_benchmark_tree_models()` | `test_m3_tree_models_and_ensembles` | `static/charts/m3_tree_benchmarks.png` | **COMPLETE** |
| **CO4.10** | Permutation Feature Importance | Out-of-sample shuffle test metric drop | [`ml/m3_trees.py`](file:///Users/hardik/RideRush/ml/m3_trees.py): `train_and_benchmark_tree_models()` | `test_m3_tree_models_and_ensembles` | `static/charts/m3_tree_benchmarks.png` | **COMPLETE** |
| **CO4.11** | SHAP Game-Theoretic Foundations | Efficiency, Symmetry, Dummy, Additivity axioms; TreeSHAP $\mathcal{O}(TLD^2)$ | [`ml/m3_trees.py`](file:///Users/hardik/RideRush/ml/m3_trees.py): `get_tree_educational_theory()` | Code & Doc inspection | `/workbench?tab=trees` | **COMPLETE** |

---

### Course Outcome 5 (CO5): Unsupervised Learning, Clustering, PCA & Anomaly Detection

> **CO5 Objective**: Execute partitioning clustering using K-Means and K-Means++ initialization, determine optimal clusters via Elbow WCSS and Silhouette analysis, compare scalable MiniBatchKMeans, perform Hierarchical Agglomerative clustering with Ward linkage and dendrogram generation, apply DBSCAN density clustering, reduce feature dimensionality via PCA with scree variance analysis, and implement Isolation Forest and One-Class SVM anomaly detection.

| Topic ID | Required Topic | Mathematical / Algorithmic Formulation | Implementation File & Function | Test Evidence | Dashboard / API Route | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CO5.1** | K-Means & K-Means++ Seeding | Distance-squared seeding: $P(x) = \frac{D(x)^2}{\sum D(x)^2}$ | [`ml/m4_unsupervised.py`](file:///Users/hardik/RideRush/ml/m4_unsupervised.py): `run_complete_m4_unsupervised()` | `test_m4_unsupervised_clustering_and_anomalies` | `/workbench?tab=unsupervised` | **COMPLETE** |
| **CO5.2** | Elbow Curve (WCSS Inertia) | $\text{WCSS}(K) = \sum_{k=1}^K \sum_{x \in C_k} \|x - \mu_k\|^2$ | [`ml/m4_unsupervised.py`](file:///Users/hardik/RideRush/ml/m4_unsupervised.py): `evaluate_kmeans_elbow_and_silhouette()` | `test_m4_unsupervised_clustering_and_anomalies` | `static/charts/m4_unsupervised_clusters.png` | **COMPLETE** |
| **CO5.3** | Silhouette Analysis | $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]$ | [`ml/m4_unsupervised.py`](file:///Users/hardik/RideRush/ml/m4_unsupervised.py): `evaluate_kmeans_elbow_and_silhouette()` | `test_m4_unsupervised_clustering_and_anomalies` | `static/charts/m4_unsupervised_clusters.png` | **COMPLETE** |
| **CO5.4** | MiniBatchKMeans Scalability | Stochastic mini-batch centroid gradient updates | [`ml/m4_unsupervised.py`](file:///Users/hardik/RideRush/ml/m4_unsupervised.py): `run_complete_m4_unsupervised()` | `test_m4_unsupervised_clustering_and_anomalies` | `/workbench?tab=unsupervised` | **COMPLETE** |
| **CO5.5** | Hierarchical Ward Linkage | Minimizing within-cluster variance increase $\Delta \text{ESS}$ | [`ml/m4_unsupervised.py`](file:///Users/hardik/RideRush/ml/m4_unsupervised.py): `run_complete_m4_unsupervised()` | `test_m4_unsupervised_clustering_and_anomalies` | `static/charts/m4_unsupervised_clusters.png` | **COMPLETE** |
| **CO5.6** | Dendrogram Tree Synthesis | Euclidean cophenetic distance hierarchy visualization | [`ml/m4_unsupervised.py`](file:///Users/hardik/RideRush/ml/m4_unsupervised.py): `create_unsupervised_plots()` | `test_m4_unsupervised_clustering_and_anomalies` | `static/charts/m4_unsupervised_clusters.png` | **COMPLETE** |
| **CO5.7** | DBSCAN Density Clustering | $\epsilon\text{-neighborhood}$, `min_samples`; Core, Border, Noise | [`ml/m4_unsupervised.py`](file:///Users/hardik/RideRush/ml/m4_unsupervised.py): `run_complete_m4_unsupervised()` | `test_m4_unsupervised_clustering_and_anomalies` | `/workbench?tab=unsupervised` | **COMPLETE** |
| **CO5.8** | Principal Component Analysis (PCA) | Covariance eigendecomposition, Scree plot, 2D projection | [`ml/m4_unsupervised.py`](file:///Users/hardik/RideRush/ml/m4_unsupervised.py): `run_complete_m4_unsupervised()` | `test_m4_unsupervised_clustering_and_anomalies` | `static/charts/m4_unsupervised_clusters.png` | **COMPLETE** |
| **CO5.9** | Isolation Forest Anomaly Detection | Recursive random axis partitioning; path length $h(x)$ | [`ml/m4_unsupervised.py`](file:///Users/hardik/RideRush/ml/m4_unsupervised.py): `run_complete_m4_unsupervised()` | `test_m4_unsupervised_clustering_and_anomalies` | `static/charts/m4_unsupervised_clusters.png` | **COMPLETE** |
| **CO5.10** | One-Class SVM & Autoencoder Preview | Non-linear RBF kernel boundary & bottleneck reconstruction | [`ml/m4_unsupervised.py`](file:///Users/hardik/RideRush/ml/m4_unsupervised.py): `run_complete_m4_unsupervised()`, `get_unsupervised_educational_theory()` | `test_m4_unsupervised_clustering_and_anomalies` | `/workbench?tab=unsupervised` | **COMPLETE** |

---

### Course Outcome 6 (CO6): Rigorous Model Evaluation, Calibration & Production ML Engineering (MLOps)

> **CO6 Objective**: Execute rigorous cross-validation protocols (5-Fold, Stratified 5-Fold, Nested 3x3 CV), benchmark Grid vs. Random hyperparameter tuning, evaluate Platt and Isotonic probability calibration with Brier scores, perform McNemar's hypothesis test and Paired Bootstrap 95% confidence interval estimation, map the 12-stage ML production lifecycle, establish a cryptographic model registry with SHA256 integrity, implement runtime input schema guardrails, serve real-time and batch predictions, monitor operational telemetry and Kolmogorov-Smirnov/PSI feature drift, enforce promotion gates, and containerize via Docker.

| Topic ID | Required Topic | Mathematical / Algorithmic Formulation | Implementation File & Function | Test Evidence | Dashboard / API Route | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CO6.1** | Comprehensive Evaluation Metrics | MAE, MSE, RMSE, $R^2$, MAPE, Residual skew, Accuracy, Precision, Recall, $F_1$, AUC, Brier | [`ml/m5_evaluation.py`](file:///Users/hardik/RideRush/ml/m5_evaluation.py): `compute_comprehensive_regression_metrics()` | `test_m5_evaluation_protocols_and_calibration` | `/workbench?tab=evaluation` | **COMPLETE** |
| **CO6.2** | 5-Fold Cross-Validation | K-Fold partition estimating generalization variance | [`ml/m5_evaluation.py`](file:///Users/hardik/RideRush/ml/m5_evaluation.py): `run_model_selection_experiments()` | `test_m5_evaluation_protocols_and_calibration` | `/workbench?tab=evaluation` | **COMPLETE** |
| **CO6.3** | Stratified 5-Fold Cross-Validation | Preserving minority target class proportions in folds | [`ml/m5_evaluation.py`](file:///Users/hardik/RideRush/ml/m5_evaluation.py): `run_model_selection_experiments()` | `test_m5_evaluation_protocols_and_calibration` | `/workbench?tab=evaluation` | **COMPLETE** |
| **CO6.4** | Nested Cross-Validation (3x3) | Outer 3-Fold evaluation, Inner 3-Fold tuning (unbiased) | [`ml/m5_evaluation.py`](file:///Users/hardik/RideRush/ml/m5_evaluation.py): `run_model_selection_experiments()` | `test_m5_evaluation_protocols_and_calibration` | `/workbench?tab=evaluation` | **COMPLETE** |
| **CO6.5** | Grid vs. Random Hyperparameter Tuning | Exhaustive lattice vs. uniform sampling runtime efficiency | [`ml/m5_evaluation.py`](file:///Users/hardik/RideRush/ml/m5_evaluation.py): `run_model_selection_experiments()` | `test_m5_evaluation_protocols_and_calibration` | `/workbench?tab=evaluation` | **COMPLETE** |
| **CO6.6** | Platt Scaling Probability Calibration | Parametric Sigmoid: $P(y=1\|f) = \frac{1}{1 + \exp(Af + B)}$ | [`ml/m5_evaluation.py`](file:///Users/hardik/RideRush/ml/m5_evaluation.py): `run_model_selection_experiments()` | `test_m5_evaluation_protocols_and_calibration` | `static/charts/m5_model_evaluation.png` | **COMPLETE** |
| **CO6.7** | Isotonic Regression Calibration | Non-parametric monotonic piecewise constant mapping | [`ml/m5_evaluation.py`](file:///Users/hardik/RideRush/ml/m5_evaluation.py): `run_model_selection_experiments()` | `test_m5_evaluation_protocols_and_calibration` | `static/charts/m5_model_evaluation.png` | **COMPLETE** |
| **CO6.8** | McNemar's Hypothesis Test | $\chi^2 = \frac{(\|b - c\| - 1)^2}{b + c} \sim \chi^2_1,\, p < 0.05$ | [`ml/m5_evaluation.py`](file:///Users/hardik/RideRush/ml/m5_evaluation.py): `compute_mcnemar_test()` | `test_m5_evaluation_protocols_and_calibration` | `/workbench?tab=evaluation` | **COMPLETE** |
| **CO6.9** | Paired Bootstrap 95% Confidence Intervals | Non-parametric $B=1000$ resamples for empirical CI | [`ml/m5_evaluation.py`](file:///Users/hardik/RideRush/ml/m5_evaluation.py): `compute_paired_bootstrap_ci()` | `test_m5_evaluation_protocols_and_calibration` | `/workbench?tab=evaluation` | **COMPLETE** |
| **CO6.10** | Learning Curves & Bias/Variance | Training vs. validation score progression across sample sizes | [`ml/m5_evaluation.py`](file:///Users/hardik/RideRush/ml/m5_evaluation.py): `create_evaluation_plots()` | `test_m5_evaluation_protocols_and_calibration` | `static/charts/m5_model_evaluation.png` | **COMPLETE** |
| **CO6.11** | 12-Stage ML Production Lifecycle | Problem scoping through to operational retirement mapping | [`ml/m1_lifecycle.py`](file:///Users/hardik/RideRush/ml/m1_lifecycle.py): `get_lifecycle_stage_mapping()` | `test_m1_lifecycle_mapping` | `/workbench?tab=lifecycle` | **COMPLETE** |
| **CO6.12** | Model Registry Artifact Persistence | Serialized binary `.joblib` paired with metadata `.json` | [`ml/m6_engineering.py`](file:///Users/hardik/RideRush/ml/m6_engineering.py): `save_model_artifact()`, `load_model_artifact()` | `test_m6_engineering_registry_and_telemetry` | `outputs/models/` | **COMPLETE** |
| **CO6.13** | SHA256 Cryptographic Verification | Tamper and corruption detection before deserialization | [`ml/m6_engineering.py`](file:///Users/hardik/RideRush/ml/m6_engineering.py): `compute_sha256()` | `test_m6_engineering_registry_and_telemetry` | `/workbench?tab=engineering` | **COMPLETE** |
| **CO6.14** | Deserialization Security Advisory | Guardrails against unverified bytecode pickle exploits | [`ml/m6_engineering.py`](file:///Users/hardik/RideRush/ml/m6_engineering.py): `save_model_artifact()` | `test_m6_engineering_registry_and_telemetry` | `/workbench?tab=engineering` | **COMPLETE** |
| **CO6.15** | Runtime Input Schema Guardrails | Strict data type coercion, range bounds, default fallback | [`ml/common.py`](file:///Users/hardik/RideRush/ml/common.py): `validate_inference_payload()` | `test_api_predict_route` | `/api/workbench/predict` | **COMPLETE** |
| **CO6.16** | Dual Prediction Serving Engines | Real-time single point & bounded batch CSV/JSON serving | [`ml/m1_lifecycle.py`](file:///Users/hardik/RideRush/ml/m1_lifecycle.py): `execute_batch_prediction()`, `app.py` | `test_m1_batch_prediction`, `test_workbench_page_and_api_routes` | `/workbench`, `/api/workbench/batch-predict` | **COMPLETE** |
| **CO6.17** | Operational Serving Telemetry | Sliding window request counts, $p50$, $p95$, $p99$ latency | [`ml/m6_engineering.py`](file:///Users/hardik/RideRush/ml/m6_engineering.py): `record_prediction_telemetry()`, `get_monitoring_telemetry()` | `test_m6_engineering_registry_and_telemetry` | `/api/workbench/telemetry`, `/workbench?tab=engineering` | **COMPLETE** |
| **CO6.18** | Feature Distribution Drift (KS & PSI) | Kolmogorov-Smirnov test & Population Stability Index | [`ml/m6_engineering.py`](file:///Users/hardik/RideRush/ml/m6_engineering.py): `detect_feature_drift()`, `calculate_psi()` | `test_m6_feature_drift_detection` | `/api/workbench/drift`, `/workbench?tab=engineering` | **COMPLETE** |
| **CO6.19** | Explicit Model Promotion Gate | Automated rule enforcing $>0.5\%$ improvement over champion | [`ml/m6_engineering.py`](file:///Users/hardik/RideRush/ml/m6_engineering.py): `evaluate_promotion_gate()` | `test_m6_engineering_registry_and_telemetry` | `/workbench?tab=engineering` | **COMPLETE** |
| **CO6.20** | Containerization & Orchestration | Multi-stage slim Docker image with curl healthchecks | [`Dockerfile`](file:///Users/hardik/RideRush/Dockerfile), [`docker-compose.yml`](file:///Users/hardik/RideRush/docker-compose.yml) | Root directory verification | CLI / Docker daemon | **COMPLETE** |

---

## Verification Summary Across All Test Suites

The entire test suite (`tests/test_app.py`) was executed against the active repository and virtual environment. All 35 tests passed cleanly with 0 failures:

```text
============================= test session starts ==============================
platform darwin -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/hardik/RideRush
plugins: anyio-4.14.2
collected 35 items

tests/test_app.py ...................................                    [100%]

============================== 35 passed in 59.21s ==============================
```
