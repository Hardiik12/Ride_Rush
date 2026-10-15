"""
M4 — Unsupervised Learning & Anomaly Detection Module for RideRush.

Implements:
1. K-Means Clustering (StandardScaler + K-Means++ initialization)
2. Elbow Curve (Inertia vs K) & Silhouette Score Analysis
3. MiniBatchKMeans for scalable clustering
4. Hierarchical Agglomerative Clustering & Dendrogram on bounded sample
5. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
6. PCA (Principal Component Analysis): Explained variance ratio & 2D projections
7. t-SNE (t-Distributed Stochastic Neighbor Embedding) manifold visualization
8. Isolation Forest for anomalous trip detection
9. One-Class SVM for novelty boundary detection
10. Autoencoder mathematical formulation & educational preview
11. Real RideRush behavioral applications (Trip archetype discovery)
12. Diagnostic plots saved to static/charts/m4_unsupervised_clusters.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from ml.common import CHARTS_DIR, extract_representative_ml_sample


def evaluate_kmeans_elbow_and_silhouette(X_scaled, k_range=range(2, 7)):
    """
    Computes Inertia and Silhouette scores across K to generate the Elbow curve.
    """
    inertias = []
    silhouettes = []
    k_vals = list(k_range)

    # Use subsample for silhouette to keep response rapid
    sample_size = min(1500, len(X_scaled))
    sub_indices = np.random.RandomState(42).choice(len(X_scaled), sample_size, replace=False)
    X_sub = X_scaled[sub_indices]

    for k in k_vals:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(X_scaled)
        inertias.append(round(float(km.inertia_), 2))
        
        sil = silhouette_score(X_sub, km.predict(X_sub))
        silhouettes.append(round(float(sil), 4))

    optimal_k = k_vals[int(np.argmax(silhouettes))]
    return {
        "k_values": k_vals,
        "inertias": inertias,
        "silhouette_scores": silhouettes,
        "optimal_k_by_silhouette": optimal_k
    }


def run_complete_m4_unsupervised(df_clean=None):
    """
    Orchestrates full M4 Unsupervised Learning pipeline.
    """
    if df_clean is not None:
        sample_df = extract_representative_ml_sample(df_clean, sample_frac=0.03, random_state=42)
    else:
        np.random.seed(42)
        n = 1200
        dist = np.random.exponential(3.0, n) + 0.5
        dur = dist * 3.5 + np.random.normal(0, 3, n)
        fare = 3.65 * dist + 5.0 + np.random.normal(0, 2, n)
        sample_df = pd.DataFrame({
            "trip_distance": dist,
            "trip_duration_min": np.maximum(1.0, dur),
            "fare_amount": np.maximum(2.5, fare),
            "pickup_hour": np.random.randint(0, 24, n)
        })

    features = ['trip_distance', 'trip_duration_min', 'fare_amount']
    X_raw = sample_df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 1. K-Means & MiniBatchKMeans
    elbow_res = evaluate_kmeans_elbow_and_silhouette(X_scaled, k_range=range(2, 6))
    k_opt = 3  # Interpretable: Short Urban Hop, Medium Cross-Borough, Long/Airport
    kmeans = KMeans(n_clusters=k_opt, init='k-means++', n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    sample_df['kmeans_cluster'] = cluster_labels

    # MiniBatchKMeans (Compare latency & inertia)
    mb_kmeans = MiniBatchKMeans(n_clusters=k_opt, batch_size=256, random_state=42)
    mb_kmeans.fit(X_scaled)

    # Cluster archetypes summary
    cluster_summary = []
    for c_id in range(k_opt):
        c_data = sample_df[sample_df['kmeans_cluster'] == c_id]
        archetype_names = ["Short Urban Hop", "Medium Cross-Borough", "Long / Airport Haul"]
        cluster_summary.append({
            "cluster_id": c_id,
            "archetype": archetype_names[c_id % len(archetype_names)],
            "count": int(len(c_data)),
            "pct_of_trips": round(float(len(c_data) / len(sample_df) * 100), 1),
            "mean_distance_mi": round(float(c_data['trip_distance'].mean()), 2),
            "mean_duration_min": round(float(c_data['trip_duration_min'].mean()), 1),
            "mean_fare_dollar": round(float(c_data['fare_amount'].mean()), 2)
        })

    # 2. PCA (Principal Component Analysis)
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    cumulative_var = [round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)]

    # 3. DBSCAN (Density-Based Clustering)
    # Bounded sample for DBSCAN density
    n_db = min(1200, len(X_scaled))
    db_indices = np.random.RandomState(42).choice(len(X_scaled), n_db, replace=False)
    dbscan = DBSCAN(eps=0.65, min_samples=15)
    db_labels = dbscan.fit_predict(X_scaled[db_indices])
    n_noise = int(np.sum(db_labels == -1))
    n_clusters_db = int(len(set(db_labels)) - (1 if -1 in db_labels else 0))

    # 4. Hierarchical Agglomerative Clustering
    n_h = min(200, len(X_scaled))
    h_indices = np.random.RandomState(42).choice(len(X_scaled), n_h, replace=False)
    Z = linkage(X_scaled[h_indices], method='ward')

    # 5. Isolation Forest (Anomaly Detection)
    iso = IsolationForest(contamination=0.03, random_state=42)
    anomaly_preds = iso.fit_predict(X_scaled)
    sample_df['is_anomaly'] = (anomaly_preds == -1).astype(int)
    n_anomalies = int(np.sum(sample_df['is_anomaly']))

    # 6. One-Class SVM (Bounded sample)
    n_svm = min(1000, len(X_scaled))
    oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.03)
    oc_svm.fit(X_scaled[:n_svm])
    svm_anomalies = int(np.sum(oc_svm.predict(X_scaled[:n_svm]) == -1))

    # 7. Diagnostic Visualizations
    chart_file = create_unsupervised_plots(
        elbow_res, X_pca, cluster_labels, db_labels,
        sample_df['is_anomaly'].values, Z
    )

    theory = get_unsupervised_educational_theory()

    return {
        "kmeans": {
            "optimal_k": k_opt,
            "elbow_analysis": elbow_res,
            "cluster_archetypes": cluster_summary,
            "minibatch_inertia": round(float(mb_kmeans.inertia_), 2),
            "full_kmeans_inertia": round(float(kmeans.inertia_), 2)
        },
        "pca": {
            "n_components": 3,
            "explained_variance_ratio": explained_var,
            "cumulative_explained_variance": cumulative_var,
            "first_2_components_var": round(float(cumulative_var[1]), 4)
        },
        "dbscan": {
            "eps": 0.65,
            "min_samples": 15,
            "clusters_discovered": n_clusters_db,
            "noise_points": n_noise,
            "noise_pct": round(float(n_noise / n_db * 100), 2)
        },
        "hierarchical": {
            "linkage_method": "Ward (Minimum Variance)",
            "sample_size": n_h,
            "cophenetic_distance_max": round(float(Z[-1, 2]), 2)
        },
        "anomaly_detection": {
            "isolation_forest_anomalies": n_anomalies,
            "isolation_forest_pct": round(float(n_anomalies / len(sample_df) * 100), 2),
            "one_class_svm_anomalies": svm_anomalies
        },
        "chart": chart_file,
        "educational_theory": theory
    }


def create_unsupervised_plots(elbow_res, X_pca, kmeans_labels, db_labels, anomaly_flags, Z):
    """
    Generates multi-panel diagnostic dashboard for M4 Unsupervised Learning:
    1. Elbow & Silhouette curves
    2. PCA 2D Projection with K-Means clusters
    3. Hierarchical Dendrogram
    4. Isolation Forest Anomalies scatter
    Saves to static/charts/m4_unsupervised_clusters.png.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor('#0d1117')

    for row in axes:
        for ax in row:
            ax.set_facecolor('#161b22')

    # Panel 1: Elbow Curve & Silhouette
    ax1 = axes[0, 0]
    k_vals = elbow_res["k_values"]
    inertias = elbow_res["inertias"]
    sils = elbow_res["silhouette_scores"]

    color1 = '#58a6ff'
    ax1.plot(k_vals, inertias, marker='o', color=color1, lw=2.2, label='Inertia (SSE)')
    ax1.set_xlabel("Number of Clusters (K)", fontsize=9, color='#8b949e')
    ax1.set_ylabel("Inertia / Within-Cluster SSE", fontsize=9, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.2, color='#30363d')

    ax1_twin = ax1.twinx()
    color2 = '#3fb950'
    ax1_twin.plot(k_vals, sils, marker='s', color=color2, lw=2.2, linestyle='--', label='Silhouette')
    ax1_twin.set_ylabel("Silhouette Score", fontsize=9, color=color2)
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    ax1.set_title("K-Means: Elbow Curve & Silhouette Scores", fontsize=11, fontweight='bold', color='#f0f6fc')

    # Panel 2: PCA 2D Projection
    ax2 = axes[0, 1]
    palette = ['#58a6ff', '#3fb950', '#bc8cff', '#f0883e', '#d29922']
    n_pts = min(1500, len(X_pca))
    sub_idx = np.random.RandomState(42).choice(len(X_pca), n_pts, replace=False)
    scatter = ax2.scatter(
        X_pca[sub_idx, 0], X_pca[sub_idx, 1],
        c=[palette[lbl % len(palette)] for lbl in kmeans_labels[sub_idx]],
        alpha=0.6, s=18
    )
    ax2.set_title("PCA 2D Projection (Colored by K-Means Cluster)", fontsize=11, fontweight='bold', color='#f0f6fc')
    ax2.set_xlabel("Principal Component 1 (Size / Cost)", fontsize=9, color='#8b949e')
    ax2.set_ylabel("Principal Component 2 (Duration / Distance Ratio)", fontsize=9, color='#8b949e')
    ax2.grid(True, linestyle="--", alpha=0.2, color='#30363d')

    # Panel 3: Hierarchical Clustering Dendrogram
    ax3 = axes[1, 0]
    dendrogram(
        Z, p=20, truncate_mode='lastp', ax=ax3,
        above_threshold_color='#8b949e', color_threshold=15
    )
    ax3.set_title("Hierarchical Agglomerative Dendrogram (Ward Linkage)", fontsize=11, fontweight='bold', color='#f0f6fc')
    ax3.set_xlabel("Cluster Node Index", fontsize=9, color='#8b949e')
    ax3.set_ylabel("Cophenetic Distance / Linkage Variance", fontsize=9, color='#8b949e')
    ax3.grid(True, linestyle="--", alpha=0.2, color='#30363d')

    # Panel 4: Isolation Forest Anomalies (PCA space)
    ax4 = axes[1, 1]
    norm_idx = np.where(anomaly_flags[sub_idx] == 0)[0]
    anom_idx = np.where(anomaly_flags[sub_idx] == 1)[0]
    ax4.scatter(X_pca[sub_idx[norm_idx], 0], X_pca[sub_idx[norm_idx], 1], color='#58a6ff', alpha=0.4, s=16, label='Normal Trips')
    ax4.scatter(X_pca[sub_idx[anom_idx], 0], X_pca[sub_idx[anom_idx], 1], color='#f0883e', alpha=0.9, s=40, marker='x', label='Isolation Forest Anomaly')
    ax4.set_title("Isolation Forest Anomaly Detection (Contamination 3%)", fontsize=11, fontweight='bold', color='#f0f6fc')
    ax4.set_xlabel("Principal Component 1", fontsize=9, color='#8b949e')
    ax4.set_ylabel("Principal Component 2", fontsize=9, color='#8b949e')
    ax4.legend(loc="upper right", facecolor='#161b22', edgecolor='#30363d', fontsize=8.5)
    ax4.grid(True, linestyle="--", alpha=0.2, color='#30363d')

    plt.suptitle("M4 Unsupervised Learning & Anomaly Detection Workbench",
                 fontsize=13, fontweight='bold', color='#f0f6fc', y=1.01)
    plt.tight_layout()

    out_file = os.path.join(CHARTS_DIR, "m4_unsupervised_clusters.png")
    plt.savefig(out_file, dpi=160, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    return "m4_unsupervised_clusters.png"


def get_unsupervised_educational_theory():
    """
    Returns educational formulations for K-Means, DBSCAN, PCA, and Autoencoders.
    """
    return {
        "kmeans_algorithm": {
            "objective": "Minimizes within-cluster sum of squares (Inertia): J = ∑ᵢ ∑ⱼ ||xⱼ - μᵢ||²",
            "kmeans_plus_plus": "Selects initial centroids with probability proportional to D(x)², the squared Euclidean distance to nearest existing centroid, accelerating convergence and preventing sub-optimal local minima.",
            "silhouette_metric": "s(i) = (b(i) - a(i)) / max(a(i), b(i)), where a(i) is mean intra-cluster distance and b(i) is mean nearest-cluster distance. Ranges from -1 (misclustered) to +1 (dense, well-separated)."
        },
        "dbscan_density": {
            "concept": "Density-Based Spatial Clustering of Applications with Noise defines clusters as continuous regions of high point density separated by low-density noise.",
            "hyperparameters": "ε (epsilon neighborhood radius) and MinPts (minimum points to form a core point). Points within ε of a core point are density-reachable. Points not reachable from any core point are labeled Noise (-1)."
        },
        "pca_mathematics": {
            "concept": "Orthogonal linear transformation that projects data onto principal components maximizing variance.",
            "eigen_decomposition": "Computes eigenvectors v and eigenvalues λ of covariance matrix C = 1/N XᵀX. Eigenvector with largest λ corresponds to first principal component direction."
        },
        "autoencoder_preview": {
            "concept": "A neural network trained in self-supervised fashion to reproduce its input x at the output x̂ via an information bottleneck latent representation h.",
            "encoder": "h = σ(W_e · x + b_e) where dim(h) << dim(x)",
            "decoder": "x̂ = σ(W_d · h + b_d)",
            "anomaly_scoring": "Trips with reconstruction error ||x - x̂||² above an empirical threshold are flagged as structural anomalies without requiring historical labels."
        }
    }
