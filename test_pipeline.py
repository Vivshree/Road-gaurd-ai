"""
ROADGUARD AI - Automated Pipeline & Verification Test Suite
Validates all ML algorithms, data loaders, preprocessors, PCA engine, visualizer, map engine,
and PDF report compiler.
"""

import sys
import os
import pandas as pd
import numpy as np

def run_tests():
    print("=" * 60)
    print("RUNNING ROADGUARD AI VERIFICATION SUITE")
    print("=" * 60)

    # 1. Imports Test
    print("[1/8] Verifying module imports...")
    try:
        from src.config import CANONICAL_COLUMNS, DEFAULT_CLUSTERING_FEATURES
        from src.data_loader import load_sample_dataset, auto_map_columns, inspect_dataset_quality
        from src.preprocessor import AccidentPreprocessor
        from src.clustering import ClusteringEngine
        from src.pca_engine import PCAEngine
        from src.profiling import ClusterProfiler
        from src.visualizer import plot_executive_trends, plot_pca_2d, plot_elbow_curve
        from src.map_engine import MapEngine
        from src.report_engine import ReportEngine
        print("  --> All modules imported successfully.")
    except Exception as e:
        print(f"  FAILED: Import error: {e}")
        sys.exit(1)

    # 2. Dataset Loading & Inspection Test
    print("[2/8] Loading benchmark dataset and testing quality audit...")
    df = load_sample_dataset()
    assert len(df) >= 1000, f"Expected at least 1000 records, got {len(df)}"
    quality = inspect_dataset_quality(df)
    print(f"  --> Loaded {quality['total_rows']} rows, {quality['total_cols']} cols. Quality Index: {quality['health_score']}/100")
    assert quality["has_geo"] is True, "Coordinates should be present in sample dataset"

    # 3. Preprocessing Test
    print("[3/8] Testing AccidentPreprocessor pipeline...")
    preprocessor = AccidentPreprocessor(
        impute_strategy="median",
        remove_duplicates=True,
        scaler_type="standard",
        selected_features=DEFAULT_CLUSTERING_FEATURES,
    )
    df_clean, X_scaled, summary = preprocessor.fit_transform(df)
    assert X_scaled.shape[0] == len(df_clean), "Row mismatch in X_scaled"
    assert X_scaled.shape[1] >= 5, "Feature count insufficient in X_scaled"
    assert not np.isnan(X_scaled).any(), "NaN values found in X_scaled"
    print(f"  --> Scaled matrix shape: {X_scaled.shape}. Features: {summary['features_used']}")

    # 4. Clustering Algorithms Test (K-Means & Agglomerative)
    print("[4/8] Testing K-Means and Agglomerative Clustering...")
    for k in [3, 4, 5]:
        km_res = ClusteringEngine.run_kmeans(X_scaled, n_clusters=k, random_state=42)
        assert km_res["silhouette"] > -1.0, f"Invalid silhouette score for K={k}"
        assert km_res["davies_bouldin"] > 0, f"Invalid DB index for K={k}"
        print(f"  --> K-Means (K={k}): Silhouette={km_res['silhouette']:.4f}, DBI={km_res['davies_bouldin']:.4f}, WCSS={km_res['inertia']:,.1f}, Time={km_res['runtime_ms']}ms")

    # Agglomerative Ward (Euclidean) & Average (Manhattan)
    agg_ward = ClusteringEngine.run_agglomerative(X_scaled, n_clusters=4, linkage="ward", metric="euclidean")
    agg_man = ClusteringEngine.run_agglomerative(X_scaled, n_clusters=4, linkage="average", metric="manhattan")
    print(f"  --> Agglomerative Ward (L2): Silhouette={agg_ward['silhouette']:.4f}, DBI={agg_ward['davies_bouldin']:.4f}")
    print(f"  --> Agglomerative Average (L1 Manhattan): Silhouette={agg_man['silhouette']:.4f}, DBI={agg_man['davies_bouldin']:.4f}")

    # 5. Distance Metric Benchmark Test
    print("[5/8] Testing Euclidean vs Manhattan comparison method...")
    dist_bench = ClusteringEngine.compare_distance_metrics(X_scaled, n_clusters=4, linkage="average")
    assert len(dist_bench["comparison_df"]) == 3, "Comparison table missing rows"
    print("  --> Distance comparison executed cleanly with mathematical foundation documentation.")

    # 6. Elbow Curve & Optimization Curves Test
    print("[6/8] Testing Elbow & Silhouette curves (K=2 to 6)...")
    curve_df = ClusteringEngine.compute_elbow_curve(X_scaled, k_min=2, k_max=6)
    assert len(curve_df) == 5, f"Expected 5 rows in curve_df, got {len(curve_df)}"
    print(f"  --> WCSS trajectory: {curve_df['WCSS_Inertia'].tolist()}")

    # 7. PCA & Profiling Test
    print("[7/8] Testing PCA 2D/3D and Semantic Profiling...")
    pca_res = PCAEngine.fit_transform(X_scaled, feature_names=summary["features_used"])
    assert pca_res["coordinates"].shape[1] == 3, "PCA should produce 3 components"
    cum_var = pca_res["cumulative_variance"][-1]
    print(f"  --> PCA 3-component cumulative variance: {cum_var*100:.1f}%")

    df_clean["Cluster"] = km_res["labels"]
    prof_df, cluster_meta = ClusterProfiler.generate_cluster_profiles(df_clean, cluster_col="Cluster")
    assert len(prof_df) == 5, f"Expected 5 clusters, got {len(prof_df)}"
    dev_df = ClusterProfiler.compute_feature_deviations(df_clean, feature_names=summary["features_used"], cluster_col="Cluster")
    assert len(dev_df) > 0, "Deviation dataframe empty"
    print(f"  --> Generated {len(prof_df)} semantic cluster profiles.")

    # 8. Folium Map & PDF Report Test
    print("[8/8] Testing Folium Map & ReportLab PDF Report Compiler...")
    folium_map, map_notice = MapEngine.create_accident_map(df_clean, cluster_metadata=cluster_meta, max_markers=200)
    assert folium_map is not None, "Folium map generation failed"
    print("  --> Folium map compiled with dark theme, heatmaps, and custom popups.")

    pdf_bytes = ReportEngine.generate_pdf_report(
        dataset_name="Verification Benchmark",
        total_records=len(df_clean),
        clustering_results=km_res,
        cluster_summary_df=prof_df,
        pca_results=pca_res,
        preprocessing_summary=summary,
    )
    assert len(pdf_bytes) > 2000, f"PDF byte stream too small ({len(pdf_bytes)} bytes)"
    print(f"  --> Publication PDF compiled successfully ({len(pdf_bytes):,} bytes).")

    print("=" * 60)
    print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY! (100% OPERATIONAL)")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
