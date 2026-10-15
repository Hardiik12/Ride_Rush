# NYC Taxi Trip Data Analysis & Machine Learning Using Dask & Gradient Descent

A modular, high-performance Big Data Exploratory Data Analysis (EDA) and Machine Learning web application built with **Flask**, **Dask**, **Pandas**, **PyArrow**, **Matplotlib**, and **Seaborn**, using the official **NYC TLC Yellow Taxi Trip Record** dataset.

---

## 📌 Project Overview
This web application provides an interactive dashboard for exploring millions of NYC Yellow Taxi trip records and predicting fare amounts using **Simple Linear Regression trained with Gradient Descent**. By leveraging **Dask DataFrame**, the application performs out-of-core chunk calculations and parallel aggregations without Out-Of-Memory (OOM) crashes.

---

## 🎯 Objective & Workflow
```text
Raw Parquet Dataset
        ↓
   Dask Loading
        ↓
 Data Understanding
        ↓
Data Quality Analysis
        ↓
   Data Cleaning
        ↓
Visual EDA Dashboard
        ↓
 CO1 Outlier Analysis (IQR, Boxplots, Winsorization)
        ↓
 CO1 Missing Value Analysis (MCAR/MAR/MNAR, Imputation, Leakage)
        ↓
 CO1 Feature Scaling (StandardScaler, MinMaxScaler, RobustScaler, Leakage)
        ↓
 CO1 Duplicate Detection & Handling (Exact vs Subset, Risks)
        ↓
Simple Linear Regression (Gradient Descent)
        ↓
  Fare Prediction Engine
```

---

## 📊 Dataset Information
- **Dataset**: NYC Yellow Taxi Trip Record (January 2025)
- **Official Source**: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Direct Parquet URL**: [yellow_tripdata_2025-01.parquet](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet)
- **Expected Local Path**: `data/yellow_tripdata_2025-01.parquet`

---

## 🛠️ Technology Stack
- **Python 3.10+**
- **Flask** (Web application server & API routes)
- **Dask** (Out-of-core parallel DataFrames & lazy evaluation)
- **Pandas** & **NumPy** (Numerical arrays & vectorised matrix ops)
- **PyArrow** (Fast Parquet I/O reader)
- **Matplotlib** & **Seaborn** (Diagnostic plot generation)
- **scikit-learn** (`KNNImputer`, `IterativeImputer` MICE)
- **HTML5**, **CSS3** (Custom dark theme styling), **Jinja2** (Dynamic templating)

---

## 📁 Project Directory Structure
```text
NYC_Taxi_BigData/ (Workspace Root: /Users/hardik/RideRush)
│
├── app.py                     # Flask Web Application routes (/ , /data, /cleaning, /eda, /insights, /prediction, /api/predict)
├── data.py                    # Data cache manager & predict_fare accessor module
├── taxi_pipeline.py           # Master pipeline orchestrator (loading -> cleaning -> eda -> outliers -> missing values -> regression)
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
│
├── data/
│   └── yellow_tripdata_2025-01.parquet # Official NYC TLC Yellow Taxi Dataset
│
├── ml/
│   ├── __init__.py
│   ├── data_loader.py         # Phase 1: Dask lazy parquet loader & metadata extraction
│   ├── data_cleaning.py       # Phase 3 & 4: Data quality inspection & logic cleaning filters
│   ├── eda.py                 # Phase 5: Univariate, temporal, fare, location, payment, correlation & outlier analyses
│   ├── outliers.py            # CO1 Syllabus: Outlier IQR analysis, boxplots & Winsorization
│   ├── missing_values.py      # CO1 Syllabus: Missing value audit, MCAR/MAR/MNAR, Imputation & Leakage
│   └── linear_regression.py   # Machine Learning: Simple Linear Regression & Gradient Descent estimator
│
├── outputs/
│   └── reports/               # Output JSON report caching area (pipeline_summary.json)
│
├── static/
│   ├── style.css              # Custom dark-theme data-analytics stylesheet
│   └── charts/                # Generated PNG visualization plots (16 charts including missing values)
│       ├── missing_values.png
│       ├── outlier_boxplots.png
│       ├── regression_fit.png
│       └── cost_convergence.png
│
├── templates/
│   ├── base.html              # Base Jinja2 layout with dark theme & sidebar navigation
│   ├── index.html             # Dashboard homepage (dataset metrics & overview cards)
│   ├── data_loading.html      # Phase 2: Data Understanding page
│   ├── cleaning.html          # Phase 3 & 4: Data Quality & Cleaning page
│   ├── eda.html               # Phase 5: EDA Dashboard, Outliers & Missing Values page
│   ├── insights.html          # EDA Key Insights page
│   └── prediction.html        # Interactive Fare Prediction & Gradient Descent page
│
└── tests/
    └── test_app.py            # Pytest unit test suite (14 passing tests)
```

---

## 🚀 Installation & Setup

1. **Navigate to Project Directory**:
   ```bash
   cd NYC_Taxi_BigData
   ```

2. **Activate Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Dataset is Present**:
   Ensure `yellow_tripdata_2025-01.parquet` is placed in `data/`.

---

## 💻 How to Run

1. **Execute Master Pipeline**:
   ```bash
   python taxi_pipeline.py
   ```

2. **Launch the Flask Web Dashboard**:
   ```bash
   python app.py
   ```
   Open your browser and navigate to: `http://127.0.0.1:5050/`

3. **Run Automated Test Suite**:
   ```bash
   python -m pytest tests/test_app.py
   ```

---

## 🌐 Web Application Pages & Routes

- **`/` (Dashboard Home)**: Key KPI cards and pipeline architecture overview.
- **`/data` (Data Understanding)**: Schema table, column descriptions, 10 raw sample rows, and Dask summary statistics.
- **`/cleaning` (Data Quality & Cleaning)**: Missing value counts table, anomaly breakdown table, cleaning rules, and retention metrics.
- **`/eda` (EDA Dashboard & CO1 Audits)**: Complete EDA dashboard with section anchor navigation, 12 EDA charts, feature boxplots (`outlier_boxplots.png`), IQR statistics table, Outlier Treatment Matrix, Missing Value Audit (`missing_values.png`), MCAR/MAR/MNAR cards, Deletion vs. Mean/Median Imputation tables, KNN & MICE Iterative Imputation demonstrations (sampled), Missing Indicators, Data Leakage Prevention rules, and Method Comparison Matrix.
- **`/insights` (EDA Key Insights)**: Calculated empirical metrics summary.
- **`/prediction` (Linear Regression & Predictor)**: Interactive fare calculator based on trip distance, displaying model formula ($\text{Fare} = w \cdot \text{Distance} + b$), slope weight $w$, intercept bias $b$, $R^2$ accuracy, RMSE, and gradient descent cost convergence plot.
- **`/api/predict?distance=X`**: REST API returning JSON predictions.

---

## 🔍 Missing Value Handling (CO1 Syllabus)

- **Audit & Visualization**: Column-wise missing counts & percentages via Dask (`static/charts/missing_values.png`).
- **Mechanisms**: MCAR, MAR, MNAR definitions with cautious taxi domain interpretations.
- **Deletion Strategies**: Analytical evaluation of listwise Row Deletion and Column Deletion.
- **Univariate Imputation**: Mean Imputation ($x_{missing} = \text{mean}(x)$) and Median Imputation ($x_{missing} = \text{median}(x)$).
- **Multivariate Imputation**: KNN Imputation (`KNNImputer`) and MICE Iterative Imputation (`IterativeImputer`) demonstrated on representative 5% sample for Big Data efficiency.
- **Missing Indicators**: Binary indicator columns (`col_missing`, 1 for missing, 0 for present).
- **Data Leakage**: Educational guidelines explaining why imputers must fit strictly on training data (`imputer.fit(X_train)`).

---

## ⚖️ Feature Scaling & Normalization (CO1 Syllabus)

- **StandardScaler (Z-Score)**: Centers at $\mu = 0$ with $\sigma = 1$; sensitive to extreme values.
- **MinMaxScaler**: Scales strictly to $[0, 1]$; highly compressed by heavy-tailed outliers.
- **RobustScaler**: Scales using $\text{Median} = 0$ and $\text{IQR} = 1$; robust to extreme taxi trips and airport surges.
- **Comparative Plot**: Multi-panel density comparison saved to `static/charts/scaling_comparison.png`.
- **Data Leakage Safeguards**: Enforces strict train/test splitting prior to scaler fitting (`scaler.fit(X_train)`).
- **Model Taxonomy**: Clear guidance on models requiring feature scaling (KNN, SVM, GD, Neural Networks, PCA) vs. scale-invariant tree ensembles (Random Forest, XGBoost, LightGBM).

---

## 📑 Duplicate Records Detection & Handling (CO1 Syllabus)

- **Exact-Row Duplicates**: Audited across all 20 columns (0 exact duplicates found, 100% unique).
- **Business-Key Subset Duplicates**: Evaluates shared pickup/dropoff times, zones, passenger counts, and trip distances (2 rows found).
- **Legitimate Trip Risks**: Explains why deleting non-exact duplicates without medallion IDs risks dropping real simultaneous commercial rides from high-density transit hubs (airports, stations).
- **Data Retention Impact**: Analytical comparison table demonstrating retention percentages without mutating the cleaned baseline of 2,816,149 records.

---

## 🤖 Simple Linear Regression with Gradient Descent Algorithm

The regression model predicts fare amount ($\hat{y}$) from trip distance ($x$):
$$\hat{y} = w \cdot x + b$$

Gradient Descent updates parameters $w$ and $b$ iteratively across 1,000 iterations with learning rate $\alpha = 0.01$:
$$w := w - \alpha \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)}) \cdot x^{(i)}$$
$$b := b - \alpha \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})$$
