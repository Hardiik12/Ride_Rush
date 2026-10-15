# RideRush — Complete M1–M6 Syllabus Coverage Audit

This ledger tracks the implementation, test verification, and interface integration of all topics across modules M1 through M6.

---

## Module M1: Machine Learning Lifecycle & Foundations

| Topic ID | Topic Name | Status | Implementation File / Function | Test Evidence | UI / API Location | Notes & Guardrails |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M1.1** | End-to-End Lifecycle Stages | **COMPLETE** | `ml/m1_lifecycle.py:get_lifecycle_stage_mapping()` | `test_m1_lifecycle_mapping` | `/workbench?tab=lifecycle` | 12 discrete production stages mapped to RideRush files |
| **M1.2** | Learning Paradigms (Supervised, Unsupervised, RL) | **COMPLETE** | `ml/m1_lifecycle.py:get_lifecycle_educational_theory()` | `test_m1_educational_theory` | `/workbench?tab=lifecycle` | Contextualized NYC Taxi fare, clustering, and dispatch examples |
| **M1.3** | Training vs Serving Separation | **COMPLETE** | `ml/common.py:prepare_train_val_test_splits()`, `app.py` | `test_workbench_training_only_leakage_prevention` | Architecture & API | Models packaged into Pipelines; inference uses cached models |
| **M1.4** | Batch Prediction Engine | **COMPLETE** | `ml/m1_lifecycle.py:execute_batch_prediction()` | `test_m1_batch_prediction` | `/workbench` (Batch form), `/api/workbench/batch-predict` | Enforces 1000 record cap; supports CSV & JSON inputs |
| **M1.5** | Training-Serving Skew & Feature Stores | **COMPLETE** | `ml/m1_lifecycle.py:get_lifecycle_educational_theory()` | `test_m1_educational_theory` | `/workbench?tab=lifecycle` | Details Covariate Shift, Leakage, and Offline/Online dual storage |
| **M1.6** | Retraining Triggers & Gates | **COMPLETE** | `ml/m1_lifecycle.py:check_retraining_triggers()` | `test_m1_retraining_trigger` | `/workbench?tab=lifecycle`, `/workbench?tab=engineering` | Triggers based on PSI $\ge 0.20$ or metric drop $\ge 5\%$ |
| **M1.7** | Lifecycle Flowchart Visualization | **COMPLETE** | `ml/m1_lifecycle.py:create_lifecycle_flow_chart()` | `test_m1_lifecycle_chart` | `static/charts/m1_lifecycle_flow.png` | Dark-themed 12-stage node architecture diagram |

---

## Module M2: Supervised Learning (Regression & Classification)

| Topic ID | Topic Name | Status | Implementation File / Function | Test Evidence | UI / API Location | Notes & Guardrails |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M2.1** | Custom Gradient Descent Baseline | **COMPLETE** | `ml/linear_regression.py:SimpleLinearRegressionGD` | `test_linear_regression_gd_model` | `/prediction`, `/workbench?tab=regression` | $J(w, b)$ cost minimization without regularization |
| **M2.2** | Normal Equation Closed-Form OLS | **COMPLETE** | `ml/regression_workbench.py:solve_normal_equation()` | `test_normal_equation` | `/workbench?tab=regression` | Analytical solution: $\theta = (X^T X)^{-1} X^T y$ |
| **M2.3** | Polynomial Regression (Degree 2) | **COMPLETE** | `ml/regression_workbench.py:train_workbench_models()` | `test_workbench_polynomial_feature_generation` | `/workbench?tab=regression` | Controlled degree to prevent Runge's oscillation |
| **M2.4** | Ridge Regression (L2 Regularization) | **COMPLETE** | `ml/regression_workbench.py:train_workbench_models()` | `test_workbench_regularization_parameters_configurable` | `/workbench?tab=regression` | Weight shrinkage via squared Euclidean norm penalty |
| **M2.5** | Lasso Regression (L1 Regularization) | **COMPLETE** | `ml/regression_workbench.py:train_workbench_models()` | `test_workbench_regularization_parameters_configurable` | `/workbench?tab=regression` | Weight sparsity via Manhattan diamond norm penalty |
| **M2.6** | Elastic Net (Combined L1 + L2) | **COMPLETE** | `ml/regression_workbench.py:train_workbench_models()` | `test_workbench_regularization_parameters_configurable` | `/workbench?tab=regression` | Balances feature selection with collinearity stability |
| **M2.7** | Binary Logistic Regression | **COMPLETE** | `ml/m2_classification.py:train_and_evaluate_classification_models()` | `test_m2_classification_models` | `/workbench?tab=classification` | Target: `is_high_tip` (tip_fraction $\ge 18\%$) |
| **M2.8** | Multinomial Logistic Regression | **COMPLETE** | `ml/m2_classification.py:train_and_evaluate_classification_models()` | `test_m2_classification_models` | `/workbench?tab=classification` | Target: `fare_tier` (Low, Medium, High fare categories) |
| **M2.9** | Sigmoid & Softmax Activations | **COMPLETE** | `ml/m2_classification.py:sigmoid()`, `softmax()` | `test_m2_activations` | `/workbench?tab=classification` | Numerically stable vectorized implementations |
| **M2.10** | Cross-Entropy Loss | **COMPLETE** | `ml/m2_classification.py:compute_cross_entropy_loss()` | `test_m2_cross_entropy` | `/workbench?tab=classification` | Log loss with probability clipping to prevent $\log(0)$ |
| **M2.11** | Class Imbalance Handling | **COMPLETE** | `ml/m2_classification.py:train_and_evaluate_classification_models()` | `test_m2_class_imbalance` | `/workbench?tab=classification` | Evaluates uniform vs `class_weight='balanced'` |
| **M2.12** | Advanced Preprocessing & OOF Target Encoding | **COMPLETE** | `ml/m2_classification.py:out_of_fold_target_encoding()` | `test_m2_target_encoding` | Module & UI | K-Fold out-of-fold encoding prevents in-sample target leakage |

---

## Module M3: Tree-Based Models & Boosting

| Topic ID | Topic Name | Status | Implementation File / Function | Test Evidence | UI / API Location | Notes & Guardrails |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M3.1** | Decision Tree Regression & Classification | **COMPLETE** | `ml/m3_trees.py:train_and_benchmark_tree_models()` | `test_m3_tree_models` | `/workbench?tab=trees` | CART implementation with squared error / Gini |
| **M3.2** | Impurity Criteria (Gini, Entropy, MSE) | **COMPLETE** | `ml/m3_trees.py:compute_tree_split_criteria()` | `test_m3_impurity_criteria` | `/workbench?tab=trees` | Mathematical formulation and information gain |
| **M3.3** | Stopping Criteria & Pre-Pruning | **COMPLETE** | `ml/m3_trees.py:train_and_benchmark_tree_models()` | `test_m3_tree_models` | `/workbench?tab=trees` | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| **M3.4** | Cost-Complexity Post-Pruning | **COMPLETE** | `ml/m3_trees.py:compute_cost_complexity_pruning_path()` | `test_m3_pruning_path` | `/workbench?tab=trees` | Extracts effective $ccp\_\alpha$ sequence |
| **M3.5** | Random Forest & OOB Evaluation | **COMPLETE** | `ml/m3_trees.py:train_and_benchmark_tree_models()` | `test_m3_tree_models` | `/workbench?tab=trees` | 100 trees; measures Out-Of-Bag $R^2$ score |
| **M3.6** | AdaBoost Regressor | **COMPLETE** | `ml/m3_trees.py:train_and_benchmark_tree_models()` | `test_m3_tree_models` | `/workbench?tab=trees` | SAMME.R adaptive error re-weighting |
| **M3.7** | Gradient Boosting Regressor | **COMPLETE** | `ml/m3_trees.py:train_and_benchmark_tree_models()` | `test_m3_tree_models` | `/workbench?tab=trees` | Residual gradient fitting with learning rate shrinkage |
| **M3.8** | LightGBM / HistGradientBoosting | **COMPLETE** | `ml/m3_trees.py:train_and_benchmark_tree_models()` | `test_m3_tree_models` | `/workbench?tab=trees` | Uses native `HistGradientBoostingRegressor` (LightGBM equivalent) |
| **M3.9** | Feature Importance (MDI vs Permutation) | **COMPLETE** | `ml/m3_trees.py:train_and_benchmark_tree_models()` | `test_m3_feature_importance` | `static/charts/m3_tree_benchmarks.png` | Side-by-side comparison of impurity decrease vs validation drop |
| **M3.10** | SHAP Interpretability Foundations | **COMPLETE** | `ml/m3_trees.py:get_tree_educational_theory()` | `test_m3_educational_theory` | `/workbench?tab=trees` | Explains Shapley axioms, TreeSHAP polynomial time algorithm |

---

## Module M4: Unsupervised Learning & Anomaly Detection

| Topic ID | Topic Name | Status | Implementation File / Function | Test Evidence | UI / API Location | Notes & Guardrails |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M4.1** | K-Means & K-Means++ Initialization | **COMPLETE** | `ml/m4_unsupervised.py:run_complete_m4_unsupervised()` | `test_m4_kmeans_clustering` | `/workbench?tab=unsupervised` | Scaled distance/duration/fare space with $D(x)^2$ sampling |
| **M4.2** | Elbow Curve & Silhouette Analysis | **COMPLETE** | `ml/m4_unsupervised.py:evaluate_kmeans_elbow_and_silhouette()` | `test_m4_kmeans_clustering` | `static/charts/m4_unsupervised_clusters.png` | Evaluates $K \in [2, 6]$; identifies optimal inertia elbow |
| **M4.3** | MiniBatchKMeans | **COMPLETE** | `ml/m4_unsupervised.py:run_complete_m4_unsupervised()` | `test_m4_kmeans_clustering` | `/workbench?tab=unsupervised` | Benchmarked against full K-Means for memory and speed |
| **M4.4** | Hierarchical Clustering & Dendrogram | **COMPLETE** | `ml/m4_unsupervised.py:run_complete_m4_unsupervised()` | `test_m4_hierarchical` | `static/charts/m4_unsupervised_clusters.png` | Agglomerative Ward linkage with cophenetic distances |
| **M4.5** | DBSCAN Density Clustering | **COMPLETE** | `ml/m4_unsupervised.py:run_complete_m4_unsupervised()` | `test_m4_dbscan` | `/workbench?tab=unsupervised` | Identifies core points, border points, and isolated noise |
| **M4.6** | Principal Component Analysis (PCA) | **COMPLETE** | `ml/m4_unsupervised.py:run_complete_m4_unsupervised()` | `test_m4_pca_dimensions` | `static/charts/m4_unsupervised_clusters.png` | Scree plot, explained variance ratio, 2D latent projection |
| **M4.7** | Isolation Forest Anomaly Detection | **COMPLETE** | `ml/m4_unsupervised.py:run_complete_m4_unsupervised()` | `test_m4_isolation_forest` | `static/charts/m4_unsupervised_clusters.png` | Recursive partitioning detecting anomalous fare-to-distance ratios |
| **M4.8** | One-Class SVM | **COMPLETE** | `ml/m4_unsupervised.py:run_complete_m4_unsupervised()` | `test_m4_one_class_svm` | `/workbench?tab=unsupervised` | RBF kernel boundary detection on bounded sample |
| **M4.9** | Autoencoder Architectural Preview | **COMPLETE** | `ml/m4_unsupervised.py:get_unsupervised_educational_theory()` | `test_m4_educational_theory` | `/workbench?tab=unsupervised` | Bottleneck latent space $h$, reconstruction loss formulation |

---

## Module M5: Evaluation, Model Selection & Calibration

| Topic ID | Topic Name | Status | Implementation File / Function | Test Evidence | UI / API Location | Notes & Guardrails |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M5.1** | Shared Regression & Residual Metrics | **COMPLETE** | `ml/m5_evaluation.py:compute_comprehensive_regression_metrics()` | `test_m5_regression_metrics` | `/workbench?tab=evaluation` | MAE, MSE, RMSE, $R^2$, MAPE, Residual skewness |
| **M5.2** | 5-Fold Cross-Validation | **COMPLETE** | `ml/m5_evaluation.py:run_model_selection_experiments()` | `test_m5_cross_validation` | `/workbench?tab=evaluation` | Mean and standard deviation across 5 folds |
| **M5.3** | Stratified K-Fold Cross-Validation | **COMPLETE** | `ml/m5_evaluation.py:run_model_selection_experiments()` | `test_m5_cross_validation` | `/workbench?tab=evaluation` | Preserves class balance ratios across validation folds |
| **M5.4** | Nested Cross-Validation (3x3) | **COMPLETE** | `ml/m5_evaluation.py:run_model_selection_experiments()` | `test_m5_nested_cv` | `/workbench?tab=evaluation` | Decouples hyperparameter tuning from performance evaluation |
| **M5.5** | GridSearchCV vs RandomizedSearchCV | **COMPLETE** | `ml/m5_evaluation.py:run_model_selection_experiments()` | `test_m5_search_benchmark` | `/workbench?tab=evaluation` | Compares search runtime vs best parameter selection |
| **M5.6** | Probability Calibration & Brier Score | **COMPLETE** | `ml/m5_evaluation.py:run_model_selection_experiments()` | `test_m5_calibration` | `static/charts/m5_model_evaluation.png` | Evaluates Uncalibrated vs Platt Sigmoid vs Isotonic Step |
| **M5.7** | McNemar's Hypothesis Test | **COMPLETE** | `ml/m5_evaluation.py:compute_mcnemar_test()` | `test_m5_mcnemar_test` | `/workbench?tab=evaluation` | Tests discordant classification error distributions ($p < 0.05$) |
| **M5.8** | Paired Bootstrap 95% Confidence Intervals | **COMPLETE** | `ml/m5_evaluation.py:compute_paired_bootstrap_ci()` | `test_m5_bootstrap_ci` | `/workbench?tab=evaluation` | 1000 resamples for non-parametric model comparison |
| **M5.9** | Learning Curves (Bias vs Variance) | **COMPLETE** | `ml/m5_evaluation.py:create_evaluation_plots()` | `test_m5_evaluation_plots` | `static/charts/m5_model_evaluation.png` | Sample size progression curve for training and validation scores |

---

## Module M6: ML Engineering & Production Operations

| Topic ID | Topic Name | Status | Implementation File / Function | Test Evidence | UI / API Location | Notes & Guardrails |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M6.1** | Model Registry Artifact Save & Load | **COMPLETE** | `ml/m6_engineering.py:save_model_artifact()`, `load_model_artifact()` | `test_m6_model_registry_save_load` | `/workbench?tab=engineering`, `outputs/models/` | Binary `.joblib` with metadata `.json` sidecar |
| **M6.2** | SHA256 Integrity Verification | **COMPLETE** | `ml/m6_engineering.py:compute_sha256()` | `test_m6_sha256_verification` | `/workbench?tab=engineering` | Detects corrupted or tampered serialized binaries |
| **M6.3** | Serialization Security Advisory | **COMPLETE** | `ml/m6_engineering.py:get_engineering_educational_theory()` | `test_m6_educational_theory` | `/workbench?tab=engineering` | Warns against unverified pickle/joblib arbitrary code execution |
| **M6.4** | Runtime Input Schema Guardrails | **COMPLETE** | `ml/common.py:validate_inference_payload()` | `test_m6_schema_validation` | `/api/workbench/predict` | Enforces data types, positive boundaries, and defaults |
| **M6.5** | Validated Prediction Serving | **COMPLETE** | `app.py:api_workbench_predict()`, `data.py` | `test_api_predict_route`, `test_workbench_routes` | `/api/predict`, `/api/workbench/predict` | Backward-compatible baseline with explicit candidate selection |
| **M6.6** | Production Serving Telemetry | **COMPLETE** | `ml/m6_engineering.py:record_prediction_telemetry()` | `test_m6_telemetry_tracking` | `/api/workbench/telemetry`, `/workbench?tab=engineering` | Live request counter and latency percentiles (p50, p95, p99) |
| **M6.7** | Feature Distribution Drift Detection | **COMPLETE** | `ml/m6_engineering.py:detect_feature_drift()` | `test_m6_drift_detection` | `/api/workbench/drift`, `/workbench?tab=engineering` | Two-sample Kolmogorov-Smirnov test and Population Stability Index |
| **M6.8** | Explicit Model Promotion Gate | **COMPLETE** | `ml/m6_engineering.py:evaluate_promotion_gate()` | `test_m6_promotion_gate` | `/workbench?tab=engineering` | Requires candidate to demonstrate $>0.5\%$ metric improvement |
| **M6.9** | Experiment Tracking Ledger | **COMPLETE** | `ml/m6_engineering.py:log_experiment_run()` | `test_m6_experiment_logging` | `outputs/reports/experiment_runs.json` | Lightweight offline experiment tracking without external daemon |
| **M6.10** | Containerization & Deployment | **COMPLETE** | `Dockerfile`, `docker-compose.yml` | File inspection | Root directory | Multi-stage slim Python 3.13 image with curl healthchecks |
