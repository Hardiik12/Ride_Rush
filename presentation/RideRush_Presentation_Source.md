# RideRush — Technical Presentation Source & Slide-by-Slide Outline

**Presentation File**: [`presentation/RideRush_Technical_Presentation.pptx`](RideRush_Technical_Presentation.pptx)  
**Total Slides**: 45 Slides (16:9 Widescreen)  
**Target Audience**: College Project Evaluators, Academic Reviewers, Technical Viva Examiners, Senior ML Engineers  
**Source Code Repository**: [https://github.com/Hardiik12/Ride_Rush](https://github.com/Hardiik12/Ride_Rush)  
**Course Outcomes Addressed**: CO1, CO2, CO3, CO4, CO5, CO6 (100% Implemented & Verified)

---

## Slide-by-Slide Technical Traceability

| Slide # | Slide Title | Primary Focus & Visuals | Source File & Function Reference | Course Outcome / Metric Evidence |
| :---: | :--- | :--- | :--- | :--- |
| **01** | **RideRush Title Slide** | Hero header, project identity, technology stack, candidate metadata | Root repository, `README.md` | All COs (Python 3.13, Flask 3.1, Dask, Scikit-Learn) |
| **02** | **Executive Summary** | 3-Pillar Architecture: Big Data, Model Zoo, MLOps | `ml/`, `app.py`, `taxi_pipeline.py` | 3.47M raw rows, 2.81M clean (81.04%), 11 models, 35/35 tests |
| **03** | **Real-World Problem Statement** | Urban transit challenges, memory crashes, sensor noise, data leakage | Domain literature, NYC TLC documentation | High data volume, OOM failure modes in Pandas |
| **04** | **Project Objectives & Scope** | 6 Core Engineering Pillars from Ingestion to Docker deployment | Project charter, `docs/CO1_CO6_COVERAGE.md` | Verification of all 6 project milestones |
| **05** | **Existing Approach vs. RideRush** | Architectural comparison table (Ad-hoc scripts vs RideRush Platform) | `docs/IMPLEMENTATION_STATUS.md` | Systematic comparison across 7 engineering dimensions |
| **06** | **Scope, Boundaries & Limitations** | Implemented boundaries, hardware scope, and optional dependency fallbacks | `docs/FINAL_IMPLEMENTATION_REPORT.md` | LightGBM/XGBoost native HistGradientBoosting, SHAP theory |
| **07** | **NYC TLC Yellow Taxi Dataset Profile** | Data provenance, file properties, Parquet Snappy compression | `data/yellow_tripdata_2025-01.parquet` | 3,475,226 raw rows, 56.4MB Parquet (>1.8GB in-memory) |
| **08** | **Dataset Schema & Feature Dictionary** | 19 raw attributes, data types, physical meanings, ML roles | `ml/common.py: FEATURE_SCHEMA`, `data_loader.py` | Full schema dictionary across regression & classification |
| **09** | **Why Big Data Techniques Are Needed** | Memory blowup analysis, Pandas GC bottlenecks vs Dask DAG | `ml/data_loader.py`, Dask architecture | Multi-threaded out-of-core chunking vs single-node OOM |
| **10** | **Data Ingestion Code Walkthrough** | Syntax-highlighted code panel with line-by-line annotations | `ml/data_loader.py: load_taxi_data()` | Column projection pruning (45% I/O reduction), PyArrow engine |
| **11** | **Master End-to-End Pipeline Architecture** | Embedded 12-stage flowchart diagram from raw file to serving | `static/charts/m1_lifecycle_flow.png`, `m1_lifecycle.py` | Complete lifecycle data flow and stage transitions |
| **12** | **Data Quality Audit & Logical Cleaning** | Vectorized cleaning predicates, invalid record counts, retention audit | `ml/data_cleaning.py: clean_taxi_data()` | 2,816,149 clean records retained (81.04% retention) |
| **13** | **Missing-Value Handling & Imputation** | MCAR/MAR taxonomy, KNN ($K=5$), MICE IterativeImputer, leakage guards | `ml/missing_values.py: audit_missing_values()` | 140,490 nulls, strict training-only fit transforms |
| **14** | **Duplicate Detection & Operational Analysis** | Full-row exact duplicates vs temporal-spatial operational collisions | `ml/duplicates.py: analyze_duplicates()` | 0 exact full-row duplicates; 56 simultaneous pickup collisions |
| **15** | **Outlier Detection & Treatment: Tukey IQR** | Tukey bounds, boxplot diagnostics, Removal vs Retention vs Winsorization | `ml/outliers.py: calculate_iqr_statistics()` | Bounds: [-$5.00, $35.00]; Winsorization at 5th/95th percentiles |
| **16** | **Feature Scaling & Leakage Safeguards** | StandardScaler, MinMaxScaler, RobustScaler formulas & leakage proofs | `ml/scaling.py: standard_scaling()` | Zero test-split statistics leakage; model dependency rules |
| **17** | **Exploratory Data Analysis: Multi-Dimensional** | 12 diagnostic charts, correlation matrix, temporal distributions | `ml/eda.py: run_eda()`, `correlation_heatmap.png` | Distance-to-fare Pearson correlation $r = 0.91$ |
| **18** | **EDA Results & Urban Mobility Insights** | 6 actionable mobility findings (rush hour, airport flat rates, tips) | `outputs/reports/pipeline_summary.json` | Evening peak 18:00–19:00; JFK flat rates at ~$70 / 17 miles |
| **19** | **Machine Learning Workflow & Target Formulation** | Continuous fare target, binary tip target, multinomial tiers, 70/15/15 | `ml/common.py: prepare_train_val_test_splits()` | Reproducible random seed=42; zero target leakage |
| **20** | **Supervised Regression: Mathematical Foundations** | Hypothesis $h_\theta(x)$, MSE loss $J(\theta)$, batch gradient updates | `ml/linear_regression.py`, mathematical theory | Cost derivatives and Normal Equation closed-form derivation |
| **21** | **Linear Regression Code Walkthrough** | Custom Batch GD implementation vs closed-form OLS; cost convergence | `ml/linear_regression.py: SimpleLinearRegressionGD` | Step size $\alpha=0.01$, 500 epochs; cost drops monotonically |
| **22** | **Regularization Paradigms: Ridge, Lasso, Elastic Net** | Mathematical penalties ($L_1, L_2$, convex combo), geometric contours | `ml/regression_workbench.py: train_workbench_models()` | Ridge $R^2 = 0.9328$; Lasso $R^2 = 0.9321$; Elastic Net $R^2 = 0.9321$ |
| **23** | **Polynomial Regression & Non-linear Curves** | Degree 2 polynomial feature expansion, non-linear pricing shelf | `ml/regression_workbench.py`, `regression_workbench_predictions.png` | Held-out MAE: $1.15, RMSE: $3.93, $R^2 = 0.9412$ |
| **24** | **Supervised Classification & Class Imbalance** | Binary tip prediction, balanced class weighting, Softmax fare tiers | `ml/m2_classification.py`, `m2_classification_curves.png` | Balanced recall: 71.04%, ROC-AUC: 0.7589; Multi-class Acc: 98.67% |
| **25** | **Tree-Based Models: CART Pruning & Boosting** | Decision Tree pre/post-pruning, Random Forest OOB, HistGradientBoosting | `ml/m3_trees.py`, `static/charts/m3_tree_benchmarks.png` | RF $R^2_{\text{OOB}} = 0.9476$; HistGradientBoosting $R^2 = 0.9477$ (310ms) |
| **26** | **Unsupervised Learning & Anomaly Detection** | $K$-Means++ ($K=3$), Silhouette, Ward Dendrogram, DBSCAN, PCA, Isolation Forest | `ml/m4_unsupervised.py`, `m4_unsupervised_clusters.png` | WCSS = 50,126; PCA retains 98.75% variance; 2,523 anomalies |
| **27** | **Model Training Pipeline Code Walkthrough** | Scikit-Learn Pipeline factory, encapsulation, zero leakage in CV | `ml/common.py`, `ml/regression_workbench.py` | Pipeline factory combining Scaler, Polynomials, and Regressor |
| **28** | **Empirical Machine Learning Benchmark Ledger** | Comprehensive comparison table of all 11 regression & tree models | `docs/FINAL_IMPLEMENTATION_REPORT.md` | Evaluated on held-out test split of 12,607 trips |
| **29** | **CO1 Deep Dive: Big Data Wrangling & Preprocessing** | Curriculum requirements, implementation files, test evidence | `ml/data_loader.py`, `ml/data_cleaning.py`, `outliers.py` | 11 unit/integration tests passed |
| **30** | **CO2 Deep Dive: Supervised Regression & Regularization** | Curriculum requirements, implementation files, test evidence | `ml/linear_regression.py`, `regression_workbench.py` | 8 unit/integration tests passed |
| **31** | **CO3 Deep Dive: Supervised Classification & Imbalance** | Curriculum requirements, implementation files, test evidence | `ml/m2_classification.py` | 3 unit/integration tests passed |
| **32** | **CO4 Deep Dive: Tree Ensembles & Explainability** | Curriculum requirements, implementation files, test evidence | `ml/m3_trees.py` | 3 unit/integration tests passed |
| **33** | **CO5 Deep Dive: Unsupervised Learning & Anomalies** | Curriculum requirements, implementation files, test evidence | `ml/m4_unsupervised.py` | 2 unit/integration tests passed |
| **34** | **CO6 Deep Dive: Evaluation Protocols & MLOps** | Curriculum requirements, implementation files, test evidence | `ml/m5_evaluation.py`, `ml/m6_engineering.py` | 8 unit/integration tests passed |
| **35** | **Production Web Application Architecture** | Decoupled layers: Client UI, Flask Server, REST APIs, Model Registry | `app.py`, `data.py`, `templates/workbench.html` | End-to-end operational software stack |
| **36** | **Interactive Dashboard & 8-Tab ML Workbench** | Complete walkthrough of all 8 workbench tabs and live simulators | `templates/workbench.html`, `static/style.css` | Lifecycle, Regression, Classification, Trees, Unsup, Eval, Eng, COs |
| **37** | **Production REST API Suite & Microservices** | Endpoint catalog: single predict, batch CSV/JSON, drift, telemetry | `app.py`: `/api/workbench/*` | 7 operational REST endpoints |
| **38** | **API Code & Request/Response Walkthrough** | Source code from `app.py`: input validation, batching, telemetry | `app.py`: `api_workbench_predict()`, `batch_predict()` | HTTP 400 guards, 1,000-record batch cap, latency tracking |
| **39** | **Automated Testing Suite: 35/35 Passing Tests** | Test suite architecture, categories, regression safeguards | `tests/test_app.py` | 35 passed, 0 failed, 100% pass rate in ~200s |
| **40** | **Production Containerization: Docker & Compose** | Multi-stage Dockerfile, image tagging, healthchecks, volume mounts | `Dockerfile`, `docker-compose.yml`, `.dockerignore` | `riderush:latest` built (1.72GB); healthcheck passes on :5050 |
| **41** | **Key Results & Measurable Achievements** | Scorecard of top numerical results across dataset, models, and MLOps | `docs/FINAL_IMPLEMENTATION_REPORT.md` | 3.47M rows, 81.04% clean, $R^2=0.9477$, 310ms, 98.75% PCA |
| **42** | **Technical Challenges & Engineering Solutions** | Real engineering obstacles (OOM, leakage, imbalance, drift, security) | Development trajectory and test ledger | Concrete solutions implemented and verified in repository |
| **43** | **Scope Limitations & Future Engineering Roadmap** | Clear demarcation of current scope vs proposed future milestones | Project roadmap | Distributed Ray/Dask, full-year data, Prometheus/Kafka |
| **44** | **Conclusion & Academic Contributions** | Summary of engineering contributions and full CO1–CO6 delivery | Final project synthesis | Bridges theoretical ML with robust production engineering |
| **45** | **Comprehensive Viva Defense: 12 Anticipated Questions** | Razor-sharp model answers for examiners and technical evaluators | Viva question bank | Deep theoretical and practical defense across all domains |

---

## Embedded Diagnostic Visual Assets

The presentation integrates the following real diagnostic charts generated from the active repository:

1. `static/charts/m1_lifecycle_flow.png` — Master 12-Stage Production ML Lifecycle (Slide 11)
2. `static/charts/outlier_boxplots.png` — Tukey IQR Boxplots: Raw vs Winsorized Fares (Slide 15)
3. `static/charts/scaling_comparison.png` — Distribution Transformations under 3 Scalers (Slide 16)
4. `static/charts/correlation_heatmap.png` — Multi-Feature Correlation Matrix (Slide 17)
5. `static/charts/cost_convergence.png` — Batch Gradient Descent Cost Convergence Curve (Slide 21)
6. `static/charts/regression_workbench_predictions.png` — Actual vs Polynomial Predicted Fares (Slide 23)
7. `static/charts/m2_classification_curves.png` — ROC-AUC and Precision-Recall Curves (Slide 24)
8. `static/charts/m3_tree_benchmarks.png` — Decision Tree Pruning Path & Feature Importance (Slide 25)
9. `static/charts/m4_unsupervised_clusters.png` — K-Means Elbow Curve, Dendrogram & 2D PCA (Slide 26)
10. `static/charts/regression_workbench_metrics.png` — Empirical Model Comparison Chart (Slide 28)
