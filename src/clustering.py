"""
ROADGUARD AI - Machine Learning Clustering Engine
Implements K-Means, Agglomerative Hierarchical Clustering, Manhattan vs Euclidean distance
comparisons, Elbow WCSS analysis, Silhouette analysis, and Davies-Bouldin index.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score

class ClusteringEngine:
    """Manages unsupervised model execution, distance metrics, and validation scoring."""

    @staticmethod
    def run_kmeans(
        X: np.ndarray,
        n_clusters: int = 4,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int = 42,
    ) -> Dict:
        """Runs K-Means clustering and returns labels, centers, inertia, validation scores, and runtime."""
        start_time = time.perf_counter()

        model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        labels = model.fit_predict(X)
        runtime_ms = round((time.perf_counter() - start_time) * 1000, 2)

        inertia = float(model.inertia_)
        sil_score = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else -1.0
        db_score = float(davies_bouldin_score(X, labels)) if len(np.unique(labels)) > 1 else -1.0
        ch_score = float(calinski_harabasz_score(X, labels)) if len(np.unique(labels)) > 1 else -1.0

        return {
            "algorithm": "K-Means",
            "model": model,
            "labels": labels,
            "cluster_centers": model.cluster_centers_,
            "n_clusters": n_clusters,
            "inertia": inertia,
            "silhouette": round(sil_score, 4),
            "davies_bouldin": round(db_score, 4),
            "calinski_harabasz": round(ch_score, 2),
            "runtime_ms": runtime_ms,
            "init": init,
            "random_state": random_state,
        }

    @staticmethod
    def run_agglomerative(
        X: np.ndarray,
        n_clusters: int = 4,
        linkage: str = "ward",
        metric: str = "euclidean",
    ) -> Dict:
        """
        Runs Agglomerative Hierarchical Clustering with valid linkage-metric configurations.
        Ward linkage mathematically requires Euclidean distance.
        Complete, Average, Single support Euclidean, Manhattan, and Cosine.
        """
        start_time = time.perf_counter()

        # Enforce mathematical constraints
        if linkage == "ward" and metric != "euclidean":
            metric = "euclidean"

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
        )
        labels = model.fit_predict(X)
        runtime_ms = round((time.perf_counter() - start_time) * 1000, 2)

        sil_score = float(silhouette_score(X, labels, metric=metric)) if len(np.unique(labels)) > 1 else -1.0
        # Davies-Bouldin uses Euclidean distances in its standard definition
        db_score = float(davies_bouldin_score(X, labels)) if len(np.unique(labels)) > 1 else -1.0
        ch_score = float(calinski_harabasz_score(X, labels)) if len(np.unique(labels)) > 1 else -1.0

        return {
            "algorithm": "Agglomerative Hierarchical",
            "model": model,
            "labels": labels,
            "n_clusters": n_clusters,
            "linkage": linkage,
            "metric": metric,
            "silhouette": round(sil_score, 4),
            "davies_bouldin": round(db_score, 4),
            "calinski_harabasz": round(ch_score, 2),
            "runtime_ms": runtime_ms,
        }

    @staticmethod
    def compare_distance_metrics(
        X: np.ndarray,
        n_clusters: int = 4,
        linkage: str = "average",
        random_state: int = 42,
    ) -> Dict:
        """
        Conducts a mathematically rigorous comparison between Euclidean and Manhattan distances.
        Standard K-Means natively minimizes sum of squared Euclidean distances (L2 norm) because
        the sample mean is the optimal centroid for L2. For L1 (Manhattan), medians are optimal.
        Here we compare Agglomerative clustering under Euclidean vs Manhattan metrics.
        """
        # 1. Agglomerative with Euclidean
        res_euclidean = ClusteringEngine.run_agglomerative(
            X, n_clusters=n_clusters, linkage=linkage, metric="euclidean"
        )

        # 2. Agglomerative with Manhattan
        res_manhattan = ClusteringEngine.run_agglomerative(
            X, n_clusters=n_clusters, linkage=linkage, metric="manhattan"
        )

        # 3. K-Means baseline (strictly Euclidean L2)
        res_kmeans = ClusteringEngine.run_kmeans(
            X, n_clusters=n_clusters, random_state=random_state
        )

        comparison_data = [
            {
                "Model & Distance": f"Agglomerative ({linkage.title()}) - Euclidean (L2)",
                "Distance Metric": "Euclidean (L2 Norm)",
                "Mathematical Formulation": "d(x,y) = sqrt(sum((x_i - y_i)^2))",
                "Silhouette Score": res_euclidean["silhouette"],
                "Davies-Bouldin Index": res_euclidean["davies_bouldin"],
                "Runtime (ms)": res_euclidean["runtime_ms"],
            },
            {
                "Model & Distance": f"Agglomerative ({linkage.title()}) - Manhattan (L1)",
                "Distance Metric": "Manhattan (L1 Cityblock)",
                "Mathematical Formulation": "d(x,y) = sum(|x_i - y_i|)",
                "Silhouette Score": res_manhattan["silhouette"],
                "Davies-Bouldin Index": res_manhattan["davies_bouldin"],
                "Runtime (ms)": res_manhattan["runtime_ms"],
            },
            {
                "Model & Distance": "K-Means (Standard) - Squared Euclidean",
                "Distance Metric": "Squared Euclidean (Centroid = Mean)",
                "Mathematical Formulation": "min sum(||x - mu_k||^2)",
                "Silhouette Score": res_kmeans["silhouette"],
                "Davies-Bouldin Index": res_kmeans["davies_bouldin"],
                "Runtime (ms)": res_kmeans["runtime_ms"],
            },
        ]

        academic_note = (
            "MATHEMATICAL FOUNDATION NOTE:\n"
            "Standard Lloyd's K-Means minimizes the Within-Cluster Sum of Squares (WCSS), which is "
            "strictly rooted in squared Euclidean distance (L2 norm). In L2 space, the cluster centroid "
            "that minimizes total variance is analytically proven to be the arithmetic mean. "
            "Standard K-Means cannot natively optimize Manhattan distance (L1 norm); attempting to use "
            "means with L1 distance produces mathematically sub-optimal centroids (the geometric median "
            "is required instead, as in K-Medians or K-Medoids / PAM). "
            "Therefore, Agglomerative Hierarchical Clustering is utilised to reliably evaluate Manhattan "
            "versus Euclidean distance structures on this roadway accident dataset."
        )

        return {
            "comparison_df": pd.DataFrame(comparison_data),
            "res_euclidean": res_euclidean,
            "res_manhattan": res_manhattan,
            "res_kmeans": res_kmeans,
            "academic_note": academic_note,
        }

    @staticmethod
    def compute_elbow_curve(
        X: np.ndarray,
        k_min: int = 2,
        k_max: int = 10,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Computes WCSS/Inertia, Silhouette, and Davies-Bouldin scores across a range of K values."""
        results = []
        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
            labels = km.fit_predict(X)
            inertia = km.inertia_
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)

            results.append({
                "K": k,
                "WCSS_Inertia": round(inertia, 2),
                "Silhouette_Score": round(sil, 4),
                "Davies_Bouldin_Index": round(db, 4),
                "Calinski_Harabasz_Index": round(ch, 2),
            })

        df_curve = pd.DataFrame(results)

        # Estimate elbow via 2nd differences
        if len(df_curve) >= 3:
            wcss = df_curve["WCSS_Inertia"].values
            diff1 = np.diff(wcss)
            diff2 = np.diff(diff1)
            # Maximum curvature index
            best_idx = np.argmax(diff2) + 1  # offset
            suggested_k = int(df_curve["K"].iloc[best_idx])
        else:
            suggested_k = 4

        return df_curve

    @staticmethod
    def compute_sample_silhouettes(X: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, float]:
        """Calculates per-sample silhouette coefficients for plotting."""
        sample_sils = silhouette_samples(X, labels)
        mean_sil = float(np.mean(sample_sils))
        return sample_sils, mean_sil
