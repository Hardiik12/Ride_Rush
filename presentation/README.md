# RideRush — Technical Presentation & Viva Defense Package

This directory contains the complete presentation package, slide generator, speaker notes, and technical defense guide for the **RideRush NYC Taxi Big Data Analytics & Machine Learning Platform**.

---

## 📁 Directory Manifest

| File | Description | Format / Tool |
| :--- | :--- | :--- |
| [`RideRush_Technical_Presentation.pptx`](RideRush_Technical_Presentation.pptx) | **Primary PowerPoint Presentation** (45 Slides, 16:9 Widescreen, Modern Dark Theme) | Microsoft PowerPoint / Keynote / Google Slides |
| [`RideRush_Presentation_Source.md`](RideRush_Presentation_Source.md) | Slide-by-slide outline, visual elements, source code paths, and CO mappings | Markdown Document |
| [`RideRush_Presentation_Notes.md`](RideRush_Presentation_Notes.md) | Comprehensive speaker script, line-by-line code explanations, and talking points | Markdown Document |
| [`RideRush_Viva_Questions.md`](RideRush_Viva_Questions.md) | 10-domain technical viva question bank with razor-sharp model answers | Markdown Document |
| [`generate_deck.py`](generate_deck.py) | Python automation script to programmatically build the 45-slide PowerPoint deck | Python 3 (`python-pptx`) |
| [`README.md`](README.md) | This documentation file | Markdown Document |

---

## 🚀 How to Regenerate the PowerPoint Deck

The PowerPoint deck can be regenerated or modified programmatically using the included Python generator script.

### 1. Prerequisites
Ensure you are in the project root directory and your Python virtual environment is active:

```bash
# Navigate to project root
cd /Users/hardik/RideRush

# Activate Python virtual environment
source .venv/bin/activate

# Verify python-pptx is installed
pip install python-pptx
```

### 2. Run the Generator Script
Execute `generate_deck.py` to build the entire 45-slide deck:

```bash
python presentation/generate_deck.py
```

**Expected Output**:
```text
Starting RideRush 45-Slide Presentation Generation...
SUCCESS: Generated 45 slides at presentation/RideRush_Technical_Presentation.pptx
```

---

## 📊 Presentation Structure Overview (45 Slides)

- **Section A: Introduction & Problem Definition (Slides 1–6)**
  - Slide 01: Hero Title Slide (Python 3.13, Flask, Dask, Scikit-Learn, Docker)
  - Slide 02: Executive Summary (3-Pillar Architecture)
  - Slide 03: Real-World Problem Statement (Urban transit scale, OOM in Pandas)
  - Slide 04: Project Objectives & Scope (6 core engineering milestones)
  - Slide 05: Existing Approach vs. RideRush (7-dimension comparison table)
  - Slide 06: Scope, Computational Boundaries & Known Limitations
- **Section B: Dataset & Big Data Processing (Slides 7–11)**
  - Slide 07: NYC TLC Yellow Taxi Dataset Profile (3.47M rows, 56.4MB Parquet)
  - Slide 08: Dataset Schema & Feature Dictionary (19 raw + derived features)
  - Slide 09: Why Big Data Techniques Are Needed (Memory expansion vs Dask DAG)
  - Slide 10: Data Ingestion Code Walkthrough (`ml/data_loader.py: load_taxi_data`)
  - Slide 11: Master End-to-End Pipeline Architecture (12-stage flowchart)
- **Section C: Data Cleaning & Exploratory Data Analysis (Slides 12–18)**
  - Slide 12: Data Quality Audit & Logical Cleaning (`ml/data_cleaning.py`, 81.04% clean)
  - Slide 13: Missing-Value Handling & Imputation Protocols (MCAR/MAR, KNN, MICE)
  - Slide 14: Duplicate Detection & Operational Analysis (0 exact dups, 56 collisions)
  - Slide 15: Outlier Detection & Treatment: Tukey IQR & Winsorization
  - Slide 16: Feature Scaling & Leakage Safeguards (Standard vs Robust Scaler)
  - Slide 17: Exploratory Data Analysis: Multi-Dimensional Diagnostics
  - Slide 18: EDA Results & Urban Mobility Insights (Peak hours, airport flat rates)
- **Section D: Machine Learning Theory & Implementation (Slides 19–28)**
  - Slide 19: Machine Learning Workflow & Target Formulation (70/15/15 splits)
  - Slide 20: Supervised Regression Mathematical Foundations (GD vs Normal Eq)
  - Slide 21: Linear Regression Code Walkthrough (`SimpleLinearRegressionGD`)
  - Slide 22: Regularization Paradigms: Ridge, Lasso & Elastic Net
  - Slide 23: Polynomial Regression & Non-linear Curves ($R^2 = 0.9412$)
  - Slide 24: Supervised Classification & Class Imbalance (Balanced tip logistic)
  - Slide 25: Tree-Based Models: CART Pruning & Ensemble Boosting
  - Slide 26: Unsupervised Learning, Clustering & Anomaly Detection ($K$-Means $K=3$)
  - Slide 27: Model Training Pipeline Code Walkthrough (Pipeline encapsulation)
  - Slide 28: Empirical Machine Learning Benchmark Ledger (Full 11-model table)
- **Section E: Course Outcomes CO1–CO6 Verification (Slides 29–34)**
  - Slide 29: CO1 Deep Dive — Big Data Wrangling, Cleaning & EDA (11 tests passed)
  - Slide 30: CO2 Deep Dive — Supervised Regression & Regularization (8 tests passed)
  - Slide 31: CO3 Deep Dive — Supervised Classification & Imbalance (3 tests passed)
  - Slide 32: CO4 Deep Dive — Tree Ensembles & Explainability (3 tests passed)
  - Slide 33: CO5 Deep Dive — Unsupervised Learning & Anomalies (2 tests passed)
  - Slide 34: CO6 Deep Dive — Model Evaluation, Calibration & MLOps (8 tests passed)
- **Section F: Application Architecture & Integration (Slides 35–40)**
  - Slide 35: Production Web Application & Microservices Architecture
  - Slide 36: Interactive Dashboard & 8-Tab ML Workbench Walkthrough
  - Slide 37: Production REST API Suite & Microservice Endpoints (7 APIs)
  - Slide 38: API Code & Request/Response Walkthrough (`app.py`)
  - Slide 39: Automated Testing Suite: 35/35 Passing Test Cases
  - Slide 40: Production Containerization: Docker & Compose (`riderush:latest`)
- **Section G: Results, Challenges & Viva Defense (Slides 41–45)**
  - Slide 41: Key Results & Measurable Technical Achievements
  - Slide 42: Technical Challenges Encountered & Engineering Solutions
  - Slide 43: Scope Limitations & Future Engineering Roadmap
  - Slide 44: Conclusion & Summary of Academic Contributions
  - Slide 45: Comprehensive Viva Defense: 12 Anticipated Questions

---

## 🎯 How to Use for Academic Evaluation & Project Review

1. **Presentation Day**:
   - Open [`RideRush_Technical_Presentation.pptx`](RideRush_Technical_Presentation.pptx) in PowerPoint, Keynote, or Google Slides in 16:9 widescreen mode.
2. **Presenter Script**:
   - Keep [`RideRush_Presentation_Notes.md`](RideRush_Presentation_Notes.md) open on a secondary display or printed as speaking notes. It gives exact word-for-word talking points and code explanations for every slide.
3. **Viva Defense**:
   - Review [`RideRush_Viva_Questions.md`](RideRush_Viva_Questions.md) before the examination. It covers 10 technical domains and 24 common examiner questions with precise mathematical and architectural answers.
4. **Code Inspection**:
   - Refer to [`RideRush_Presentation_Source.md`](RideRush_Presentation_Source.md) to quickly jump to exact source file lines in the repository during deep-dive questioning.
