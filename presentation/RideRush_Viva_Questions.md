# RideRush — Comprehensive Technical Viva Defense Guide

**Purpose**: Razor-sharp, technically rigorous questions and model answers prepared for academic review, project viva voce, external examiners, and senior engineering evaluations.  
**Project**: RideRush — NYC Yellow Taxi Big Data Analytics & Production Machine Learning Platform  
**Repository**: [https://github.com/Hardiik12/Ride_Rush](https://github.com/Hardiik12/Ride_Rush)

---

## Domain 1: Big Data Ingestion & Architecture (CO1)

### Q1.1: Why did you use Dask instead of standard Pandas for data ingestion?
**Model Answer**:
"Standard Pandas uses **eager evaluation** and requires contiguous memory allocation. When reading our 56.4 MB compressed Parquet file (3,475,226 rows), it expands to over 1.8 GB in memory. Creating intermediate copies during filtering and cleaning spikes RAM usage beyond 8 GB, causing Out-Of-Memory (OOM) kernel crashes on typical hardware. 

Dask solves this through **lazy evaluation and blocked out-of-core execution**. It divides the dataset into discrete chunks (partitions) that fit into CPU cache, constructs a Directed Acyclic Graph (DAG) of transformations, and executes them in parallel across multi-core processors only when `.compute()` is invoked. This keeps memory consumption constant regardless of dataset size."

### Q1.2: Why did you specify `engine="pyarrow"` and `columns=[...]` in `ml/data_loader.py`?
**Model Answer**:
"In `ml/data_loader.py`, specifying `engine="pyarrow"` leverages Apache Arrow's C++ columnar parsing engine, which uses SIMD vectorization to decode Snappy-compressed Parquet blocks directly. 

Specifying `columns=[...]` enforces **column projection pruning**: since Parquet is a columnar storage format, the disk reader only reads byte ranges corresponding to the selected 11 modeling columns, completely bypassing unneeded metadata columns. This reduced disk I/O and memory overhead by approximately 45%."

---

## Domain 2: Data Cleaning & Preprocessing (CO1)

### Q2.1: Explain your logical cleaning filters. Why did you retain 81.04% of the dataset?
**Model Answer**:
"In `ml/data_cleaning.py`, our vectorized predicate filter applied five domain constraints:
1. `passenger_count` between 1 and 6 (eliminating nulls and impossible occupancies).
2. `trip_distance` between 0.01 and 100.0 miles (removing stationary records and odometer errors).
3. `fare_amount` between $2.50 (minimum flag-drop rate) and $500.00 (removing negative fares and typo entries).
4. `trip_duration` between 1.0 and 180.0 minutes (removing inverted timestamps and multi-day errors).
5. `total_amount` strictly positive.

Out of 3,475,226 raw rows, 659,077 dirty records were excluded, retaining exactly 2,816,149 verified clean records (an 81.04% retention rate)."

### Q2.2: What is the difference between MCAR, MAR, and MNAR missing data? Which did you identify in RideRush?
**Model Answer**:
"The missing data taxonomy classifies missingness into three mechanisms:
- **MCAR (Missing Completely At Random)**: The probability of missingness is completely independent of both observed and unobserved data (e.g., random network packet loss).
- **MAR (Missing At Random)**: The probability of missingness depends on observed features but not the missing value itself.
- **MNAR (Missing Not At Random)**: The probability of missingness depends directly on the unobserved value itself (e.g., passengers with very high fares refusing to disclose tips).

In RideRush (`ml/missing_values.py`), we identified 140,490 null records in `passenger_count` and `RatecodeID` and classified them as **MAR**, correlated with mobile wireless dispatch terminal hardware differences across taxi fleets."

### Q2.3: Why is fitting an imputer or scaler on the entire dataset considered a catastrophic data leakage error?
**Model Answer**:
"Fitting an imputer (`imputer.fit(X)`) or scaler (`scaler.fit(X)`) across the whole dataset before splitting causes **data leakage**. The transformer computes global statistics—such as the mean $\mu$, standard deviation $\sigma$, or neighbor topology—using test-set samples. 

When those parameters are used to transform the training split, information from the test set has secretly leaked into the training data. This causes the model to appear falsely optimistic during training and validation, leading to severe performance drops when deployed on real unseen production data. In RideRush, we strictly enforce `imputer.fit(X_train)` and `imputer.transform(X_test)`."

---

## Domain 3: Statistical Outlier Analysis (CO1)

### Q3.1: What is the Tukey IQR method and how were bounds computed in RideRush?
**Model Answer**:
"The Tukey Interquartile Range (IQR) method calculates the spread of the middle 50% of the distribution:
$$\text{IQR} = Q_3 - Q_1$$
The statistical outlier boundaries are:
$$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

In RideRush (`ml/outliers.py`), on `fare_amount`: $Q_1 = \$9.50$, $Q_3 = \$21.50$, giving $\text{IQR} = \$12.00$. The lower bound evaluates to $\$9.50 - 1.5(\$12) = -\$5.00$ (clamped at the $\$2.50$ legal minimum fare). The upper bound evaluates to $\$21.50 + 1.5(\$12) = \$35.00$."

### Q3.2: Why did you choose Winsorization over Outlier Removal or Outlier Retention?
**Model Answer**:
"We evaluated all three strategies:
1. **Removal**: Dropping all fares above $\$35.00$ discarded 126,300 trips (4.5% of clean data). This severely biased the model against legitimate long-distance and airport trips, causing large under-prediction errors on high-fare trips.
2. **Retention**: Retaining raw fares up to $\$500.00$ preserved revenue reality but skewed unregularized gradient descent weights due to squared-error sensitivity.
3. **Winsorization**: Capping top and bottom 5% values at the 5th ($\$6.00$) and 95th ($\$52.00$) percentiles retained 100% of the data volume while neutralizing extreme gradient destabilization. In empirical testing, Winsorization yielded the lowest overall generalization RMSE on long-haul trips."

---

## Domain 4: Supervised Regression & Regularization (CO2)

### Q4.1: Explain the mathematical difference between your custom Batch Gradient Descent and the Normal Equation.
**Model Answer**:
"In `ml/linear_regression.py`, **Batch Gradient Descent** is an iterative numerical optimization algorithm that minimizes Mean Squared Error cost $J(\theta)$:
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$$
It calculates partial derivatives over all $m$ samples and updates weights iteratively:
$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$
It requires tuning a learning rate $\alpha$ and running multiple epochs.

In contrast, the **Normal Equation** in `ml/regression_workbench.py` solves for the analytical global minimum in closed form by setting the gradient $\nabla_\theta J = 0$:
$$\theta = (X^T X)^{-1} X^T y$$
It requires no learning rate, no iterations, and finds the exact Ordinary Least Squares minimum in $\mathcal{O}(d^3)$ time, where $d$ is feature dimension."

### Q4.2: How do Ridge, Lasso, and Elastic Net regularizers differ mathematically and geometrically?
**Model Answer**:
"Regularization constrains model weights to prevent overfitting and handle multicollinearity:
- **Ridge ($L_2$)**: Adds an $\alpha \|\mathbf{w}\|_2^2 = \alpha \sum w_j^2$ penalty. Geometrically, its penalty region is a hypersphere. The loss contours contact the sphere smoothly along the curves, shrinking coefficients smoothly toward zero but never setting any strictly to zero.
- **Lasso ($L_1$)**: Adds an $\alpha \|\mathbf{w}\|_1 = \alpha \sum |w_j|$ penalty. Geometrically, its penalty region is a diamond/polytope. The loss contours frequently intersect the sharp corners on the coordinate axes, driving non-informative feature weights strictly to zero (performing automatic feature selection).
- **Elastic Net**: Combines both penalties: $\alpha [\rho \|\mathbf{w}\|_1 + \frac{1-\rho}{2} \|\mathbf{w}\|_2^2]$. When features are correlated (like distance and duration), Lasso arbitrarily selects one; Elastic Net groups them and stabilizes coefficient shrinkage."

### Q4.3: Why did Polynomial Regression outperform standard Linear Regression?
**Model Answer**:
"In `ml/regression_workbench.py`, Polynomial Regression expanded features to degree 2 using `PolynomialFeatures(degree=2)`. Taxi pricing is not purely linear across all distances: short trips are governed by initial flag-drop surcharges ($4.50+), medium trips scale with distance and time, and long-distance airport trips cluster around flat rates ($70.00). 

The quadratic feature expansion $[x_1, x_2, x_1^2, x_2^2, x_1 x_2]$ captured this non-linear curvature, boosting test $R^2$ from 0.8842 (Linear GD) to **0.9412** and reducing RMSE to **$\$3.93$**."

---

## Domain 5: Supervised Classification & Imbalance (CO3)

### Q5.1: How did you handle the 82.2% class imbalance in high-tip prediction?
**Model Answer**:
"In `ml/m2_classification.py`, we predicted `is_high_tip` ($\ge 20\%$). High tips represent only 17.8% of trips, creating an 82.2% negative majority. A standard unweighted logistic regression achieved 74.3% accuracy simply by predicting negative, but suffered a dismal Recall of 11.2% on high-tip trips.

We resolved this by setting `class_weight='balanced'`. Scikit-Learn weights each class inversely proportional to its training frequency:
$$w_k = \frac{N}{K \cdot n_k}$$
This heavily penalizes false negatives on the high-tip minority class, boosting high-tip Recall from 11.2% to **71.04%** and improving ROC-AUC to **0.7589**."

### Q5.2: What is Out-of-Fold (OOF) Target Encoding and why is it superior to naive label encoding?
**Model Answer**:
"`PULocationID` contains 265 distinct geographic zones. One-hot encoding would add 265 sparse columns, inflating model dimensionality. Naive target encoding replaces each location ID with the global mean target value, but this causes severe target leakage because the target of a sample is included in its own feature value.

**Out-of-Fold (OOF) Target Encoding** splits training data into $K=5$ folds. For each fold, location target means are computed strictly using the *other 4 folds*, plus m-estimate smoothing toward the global mean:
$$\hat{S}_i = \frac{n_i \cdot \bar{y}_i + m \cdot \bar{y}_{\text{global}}}{n_i + m}$$
This allows tree and linear models to leverage spatial pricing patterns with zero target leakage."

---

## Domain 6: Tree Ensembles & Boosting (CO4)

### Q6.1: What is the difference between pre-pruning and cost-complexity post-pruning in Decision Trees?
**Model Answer**:
"In `ml/m3_trees.py`:
- **Pre-pruning (Early Stopping)** halts tree growth during training using stopping criteria like `max_depth=5`, `min_samples_split=20`, or `min_samples_leaf=10`. It is computationally fast but can prematurely stop splitting before discovering complex interactions.
- **Cost-Complexity Post-Pruning (Minimal Cost-Complexity Pruning)** grows a full unconstrained tree first, then prunes branches back by minimizing the cost-complexity criterion:
$$R_\alpha(T) = R(T) + \alpha |T|$$
where $R(T)$ is training error, $|T|$ is the number of terminal leaves, and $\alpha \ge 0$ is the complexity parameter. By extracting the effective $\alpha$ sequence (`ccp_alpha`), we identify the sub-tree that maximizes validation $R^2$."

### Q6.2: What is Out-of-Bag (OOB) evaluation in Random Forest? What score did RideRush achieve?
**Model Answer**:
"In Random Forest, each tree is trained on a bootstrap sample drawn with replacement from the training set. Mathematically, the probability of a sample *not* being selected for a tree is:
$$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 36.8\%$$
These unselected samples are **Out-of-Bag (OOB)** for that tree.

By aggregating predictions for each sample using only the trees where that sample was out-of-bag, Random Forest computes an unbiased generalization score during training without requiring a separate validation split. In RideRush (`ml/m3_trees.py`), our 100-tree Random Forest achieved an **$R^2_{\text{OOB}} = 0.9476$**, matching our held-out test score of $0.9436$."

### Q6.3: Why was Histogram Gradient Boosting 15x faster than standard Gradient Boosting?
**Model Answer**:
"Traditional Gradient Boosting evaluates all possible split points across all continuous feature values at every node, requiring continuous sorting in $\mathcal{O}(N \log N)$ time.

**Histogram Gradient Boosting** (`HistGradientBoostingRegressor`) discretizes continuous features into 255 integer buckets (bins) during preprocessing. Evaluating splits then requires only computing a 255-bin histogram in $\mathcal{O}(N)$ time. Combined with leaf-wise (best-first) tree growth, it trained in just **310.9 milliseconds** compared to **4,837.1 milliseconds** for standard Gradient Boosting, while achieving our highest accuracy: $R^2 = \mathbf{0.9477}$ and $\text{RMSE} = \mathbf{\$3.65}$."

---

## Domain 7: Unsupervised Learning & Anomaly Detection (CO5)

### Q7.1: How did you select $K=3$ for $K$-Means clustering?
**Model Answer**:
"In `ml/m4_unsupervised.py`, we evaluated $K$-Means across $K \in [2, 6]$ using two complementary methods:
1. **The Elbow Method**: We plotted Within-Cluster Sum of Squares (WCSS / Inertia). A clear inflection elbow appeared at $K=3$ ($\text{WCSS} = 50,126$), after which inertia reductions diminished.
2. **Silhouette Analysis**: Silhouette scores measure how close each point is to its own cluster compared to neighboring clusters. $K=3$ achieved the highest average silhouette score of **0.6134**.

The three clusters correspond to natural transit archetypes:
- *Cluster 0 (73.8%)*: Short Urban Hops (mean 1.4 mi, $10.63).
- *Cluster 1 (20.0%)*: Airport / JFK Hauls (mean 5.4 mi, $28.29).
- *Cluster 2 (6.2%)*: Long Cross-Borough Trips (mean 16.7 mi, $67.37)."

### Q7.2: What is the purpose of PCA in RideRush? What variance did you retain?
**Model Answer**:
"In `ml/m4_unsupervised.py`, Principal Component Analysis (PCA) performs orthogonal linear transformation to project our continuous feature space onto orthogonal axes of maximum variance.
- Principal Component 1 ($PC1$) captured **91.52%** of total variance.
- Principal Component 2 ($PC2$) captured **7.23%** of total variance.

Together, the 2D projection retained **98.75% of cumulative multi-feature variance**, allowing us to visualize trip clusters and detect spatial anomalies in a 2D plane with virtually zero information loss."

### Q7.3: How does Isolation Forest detect taxi anomalies?
**Model Answer**:
"Isolation Forest operates on the principle that anomalies are 'few and different'. Instead of modeling normal cluster densities, it isolates points by randomly selecting a feature and randomly selecting a split value between the feature's minimum and maximum. 

Because anomalous trips (e.g., $\$100$ fare for a $0.5$-mile trip) occupy sparse regions of feature space, they require noticeably fewer random splits to isolate than normal trips. In RideRush, Isolation Forest isolated **2,523 structural anomalies** ($3.0\%$ contamination threshold), detecting meter tampering and driver detour fraud."

---

## Domain 8: Evaluation Protocols & Calibration (CO6)

### Q8.1: Why did you use Nested Cross-Validation instead of standard Cross-Validation?
**Model Answer**:
"Standard $K$-Fold cross-validation optimizes hyperparameters and evaluates model performance on the same validation folds. This introduces **hyperparameter selection bias** (optimistic evaluation bias), because the model architecture was chosen specifically because it scored highest on those validation folds.

**Nested Cross-Validation** uses two concentric loops:
- *Inner Loop ($3$ Folds)*: Tunes hyperparameters and performs feature selection.
- *Outer Loop ($3$ Folds)*: Evaluates the tuned model on completely untouched test folds.

In `ml/m5_evaluation.py`, Nested $3\times3$ CV yielded an outer generalization error of **$\text{RMSE} = \$4.08$**, proving that our reported model performance is completely free from hyperparameter tuning bias."

### Q8.2: What is probability calibration and why does it matter?
**Model Answer**:
"Standard classification algorithms output probability-like scores $\hat{p} \in [0, 1]$, but these scores are often not true probabilities (e.g., Logistic Regression with balanced weights or tree ensembles can be overconfident or underconfident). A model is **calibrated** if, among samples predicted with probability $p=0.80$, exactly $80\%$ are truly positive.

In `ml/m5_evaluation.py`, we evaluated Platt Sigmoid scaling and non-parametric Isotonic Regression using the **Brier Score** ($B = \frac{1}{N} \sum (p_i - y_i)^2$). Platt calibration reduced the Brier score from $0.2314$ down to **$0.2139$** (a $7.56\%$ error reduction), providing reliable probabilities for downstream dynamic dispatch systems."

### Q8.3: What is the difference between an $R^2$ score and accuracy?
**Model Answer**:
"**Accuracy** is strictly a classification metric representing the fraction of correct predictions ($\frac{\text{TP} + \text{TN}}{N}$). It has no meaning in regression.

**Coefficient of Determination ($R^2$)** measures the proportion of variance in the dependent variable explained by the regression model:
$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
$R^2 = 1.0$ represents a perfect fit; $R^2 = 0.0$ means the model performs no better than predicting the mean $\bar{y}$; and $R^2$ can be **negative** if the model performs worse than the horizontal mean line."

---

## Domain 9: Production MLOps & Architecture (CO6)

### Q9.1: Why did you implement SHA-256 verification in the Model Registry?
**Model Answer**:
"Python model serialization libraries like `joblib` and `pickle` reconstruct objects by executing Python bytecode during deserialization. If an unauthorized actor modifies or swaps a model binary on disk or during a network transfer, calling `joblib.load()` executes arbitrary attacker code with server privileges (CWE-502 vulnerability).

In `ml/m6_engineering.py`, whenever a model is saved, we compute its 64-character SHA-256 cryptographic digest and store it in a companion JSON sidecar. Before loading any binary into serving memory, `load_model_artifact()` recalculates the SHA-256 hash. If even a single byte has been modified, execution aborts with an integrity exception."

### Q9.2: How does your feature drift detection work in production?
**Model Answer**:
"In `ml/m6_engineering.py`, we monitor live serving data against our baseline training distribution using two statistical techniques:
1. **Two-Sample Kolmogorov-Smirnov (KS) Test**: Evaluates whether the cumulative distribution functions $F_{\text{ref}}(x)$ and $F_{\text{live}}(x)$ differ significantly:
$$D = \sup_x |F_{\text{ref}}(x) - F_{\text{live}}(x)|$$
If $p < 0.05$, a statistically significant distribution shift has occurred.
2. **Population Stability Index (PSI)**:
$$\text{PSI} = \sum_{k=1}^K (P_k - Q_k) \ln\left(\frac{P_k}{Q_k}\right)$$
$\text{PSI} < 0.1$ indicates no drift; $\text{PSI} \ge 0.2$ triggers an automated alert recommending model retraining."

---

## Domain 10: Course Outcome (CO1–CO6) Traceability

### Q10.1: Briefly summarize where each of CO1 through CO6 is implemented in RideRush.
**Model Answer**:
"The entire RideRush repository is mapped to Course Outcomes CO1 through CO6:
- **CO1 (Big Data & Cleaning)**: Implemented in `ml/data_loader.py`, `data_cleaning.py`, `outliers.py`, `missing_values.py`, `scaling.py`, and `duplicates.py`. Covers Dask ingestion, Tukey bounds, Winsorization, and 12 EDA charts.
- **CO2 (Regression & Regularization)**: Implemented in `ml/linear_regression.py` (custom Batch GD) and `ml/regression_workbench.py` (Normal Equation, Polynomial, Ridge, Lasso, Elastic Net).
- **CO3 (Classification & Imbalance)**: Implemented in `ml/m2_classification.py`. Covers Binary Logistic (balanced weights), Multinomial Softmax, and Out-of-Fold target encoding.
- **CO4 (Tree Ensembles & Boosting)**: Implemented in `ml/m3_trees.py`. Covers CART pruning paths, Random Forest (OOB $R^2 = 0.9476$), HistGradientBoosting ($R^2 = 0.9477$), and Permutation Importance.
- **CO5 (Unsupervised & Anomalies)**: Implemented in `ml/m4_unsupervised.py`. Covers $K$-Means++ ($K=3$), Silhouette analysis, Ward dendrogram, DBSCAN, 2D PCA (98.75% variance), and Isolation Forest.
- **CO6 (Evaluation & MLOps)**: Implemented in `ml/m5_evaluation.py` and `ml/m6_engineering.py`. Covers 5-Fold/Nested CV, Platt calibration, McNemar test, Bootstrap CIs, SHA-256 Model Registry, Telemetry, and Docker."
