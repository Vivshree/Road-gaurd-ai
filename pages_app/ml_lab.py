"""
ROADGUARD AI - Machine Learning Laboratory
Academic flagship module implementing K-Means, Agglomerative Hierarchical Clustering,
Euclidean vs Manhattan distance analysis, Elbow WCSS, Silhouette analysis, Davies-Bouldin Index,
2D/3D PCA visualization, and comprehensive model benchmarking.
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.clustering import ClusteringEngine
from src.pca_engine import PCAEngine
from src.visualizer import (
    plot_pca_2d, plot_pca_3d, plot_elbow_curve, plot_silhouette_curve,
    plot_davies_bouldin_curve, plot_sample_silhouettes
)
from src.profiling import ClusterProfiler

def render_ml_lab():
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 22px; font-weight: 800; color: #FFFFFF; margin-bottom: 4px;">
            Machine Learning Laboratory &amp; Academic Benchmark
        </h2>
        <div style="color: #94A3B8; font-size: 13.5px;">
            Rigorous unsupervised learning experimental suite. Compare clustering algorithms, evaluate mathematical distance formulations,
            and inspect 2D/3D Principal Component projections.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.get("df_processed", None)
    X_scaled = st.session_state.get("X_scaled", None)

    if df is None or X_scaled is None:
        st.warning("No preprocessed dataset available. Please visit the Dataset Upload Center first.")
        return

    # Initialize model experiment registry in session_state
    if "experiment_history" not in st.session_state:
        st.session_state["experiment_history"] = []

    # Academic Laboratory Tabs
    tab_sandbox, tab_distance, tab_curves, tab_pca, tab_benchmarks = st.tabs([
        "🔬 Algorithm Sandbox",
        "📐 Euclidean vs. Manhattan Distance",
        "📉 Validation Curves (Elbow & Silhouette)",
        "🌌 PCA Dimensionality (2D & 3D)",
        "📋 Model Comparison Matrix",
    ])

    # ==========================================
    # TAB 1: ALGORITHM SANDBOX
    # ==========================================
    with tab_sandbox:
        st.markdown("### Unsupervised Algorithm Configuration")
        algo_choice = st.radio(
            "Select Clustering Algorithm",
            options=["K-Means Clustering", "Agglomerative Hierarchical Clustering"],
            horizontal=True,
        )

        p_col1, p_col2, p_col3 = st.columns(3)

        if algo_choice == "K-Means Clustering":
            with p_col1:
                k_val = st.slider("Clusters (K)", min_value=2, max_value=10, value=st.session_state.get("cluster_k", 4), key="km_k")
                init_val = st.selectbox("Centroid Initialization", options=["k-means++", "random"], index=0, key="km_init")
            with p_col2:
                n_init_val = st.slider("n_init (Random Restarts)", min_value=5, max_value=25, value=10, key="km_n_init")
                max_iter_val = st.slider("Max Iterations", min_value=100, max_value=500, value=300, step=50, key="km_max_iter")
            with p_col3:
                seed_val = st.number_input("Reproducible Random Seed", min_value=0, max_value=9999, value=42, key="km_seed")
                st.caption("Standard Lloyd's K-Means minimizing sum of squared Euclidean deviations ($L_2$).")

            if st.button("🚀 Fit K-Means Model", use_container_width=True):
                with st.spinner("Fitting K-Means model..."):
                    res = ClusteringEngine.run_kmeans(
                        X_scaled, n_clusters=k_val, init=init_val, n_init=n_init_val,
                        max_iter=max_iter_val, random_state=seed_val
                    )
                    _log_experiment("K-Means", f"K={k_val}, init={init_val}, seed={seed_val}", res)
                    st.session_state["active_lab_model"] = res

        else:  # Agglomerative
            with p_col1:
                k_val = st.slider("Clusters (K)", min_value=2, max_value=10, value=st.session_state.get("cluster_k", 4), key="agg_k")
                linkage_val = st.selectbox("Linkage Criterion", options=["ward", "complete", "average", "single"], index=0, key="agg_linkage")
            with p_col2:
                metric_options = ["euclidean"] if linkage_val == "ward" else ["euclidean", "manhattan", "cosine"]
                metric_val = st.selectbox("Distance Metric", options=metric_options, index=0, key="agg_metric")
                if linkage_val == "ward":
                    st.caption("⚠️ **Ward Linkage constraint:** Mathematically restricted strictly to Euclidean distance ($L_2$).")
            with p_col3:
                st.caption(f"Hierarchical bottom-up tree partitioning under `{linkage_val}` linkage and `{metric_val}` metric.")

            if st.button("🚀 Fit Agglomerative Model", use_container_width=True):
                with st.spinner(f"Fitting Agglomerative Clustering (Linkage: {linkage_val}, Metric: {metric_val})..."):
                    res = ClusteringEngine.run_agglomerative(
                        X_scaled, n_clusters=k_val, linkage=linkage_val, metric=metric_val
                    )
                    _log_experiment("Agglomerative", f"K={k_val}, link={linkage_val}, metric={metric_val}", res)
                    st.session_state["active_lab_model"] = res

        # Active Model Results Display
        active_res = st.session_state.get("active_lab_model", st.session_state.get("clustering_results", None))
        if active_res:
            st.markdown("---")
            st.markdown(f"#### Model Evaluation: {active_res['algorithm']} (K = {active_res['n_clusters']})")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Silhouette Score", f"{active_res['silhouette']:.4f}", help="Cohesion vs Separation [-1 to +1]. Higher is better.")
            m2.metric("Davies-Bouldin Index", f"{active_res['davies_bouldin']:.4f}", help="Cluster similarity. Lower is better.")
            m3.metric("Calinski-Harabasz", f"{active_res['calinski_harabasz']:.1f}", help="Variance ratio criterion. Higher is better.")
            m4.metric("Convergence Latency", f"{active_res['runtime_ms']} ms", help="Model execution time.")

            # Sample Silhouette Profile Plot
            sample_sils, mean_sil = ClusteringEngine.compute_sample_silhouettes(X_scaled, active_res["labels"])
            cluster_meta = st.session_state.get("cluster_metadata", {})
            fig_sil_samples = plot_sample_silhouettes(sample_sils, active_res["labels"], cluster_meta)
            st.plotly_chart(fig_sil_samples, use_container_width=True)

    # ==========================================
    # TAB 2: EUCLIDEAN VS MANHATTAN
    # ==========================================
    with tab_distance:
        st.markdown("### Mathematical Formulation: Euclidean ($L_2$) vs. Manhattan ($L_1$) Distance")
        
        st.markdown("""
        <div class="academic-callout">
            <strong>Theoretical Principle &mdash; K-Means Distance Restriction:</strong><br/>
            Standard Lloyd's K-Means clustering is formulated around minimizing the <strong>Within-Cluster Sum of Squares (WCSS)</strong>:
            <br/><br/>
            $$\\text{WCSS} = \\sum_{k=1}^K \\sum_{x_i \\in C_k} \\|x_i - \\mu_k\\|_2^2$$
            <br/>
            In squared Euclidean space ($L_2^2$), setting the derivative of total error with respect to the centroid coordinate to zero analytically proves that the <strong>arithmetic mean</strong> is the unique global minimum:
            $$\\mu_k = \\frac{1}{|C_k|} \\sum_{x_i \\in C_k} x_i$$
            <br/>
            <strong>Crucial Academic Fact:</strong> Standard K-Means does <strong>NOT</strong> natively optimize Manhattan ($L_1$) distance ($d_1(x, y) = \\sum |x_i - y_i|$). 
            If Manhattan distance were used with cluster means, the resulting centroids would be mathematically inconsistent with the error criterion (minimizing $L_1$ requires the <strong>coordinate-wise median</strong> or medoid, as formulated in K-Medians or K-Medoids / PAM).
            <br/><br/>
            Therefore, in ROADGUARD AI, we evaluate Manhattan distance rigorously using <strong>Agglomerative Hierarchical Clustering</strong>, which naturally supports pairwise Manhattan metric matrices.
        </div>
        """, unsafe_allow_html=True)

        k_dist = st.slider("Clusters (K) for Distance Benchmark", min_value=2, max_value=8, value=4, key="dist_k")
        link_dist = st.selectbox("Linkage for Comparison", options=["average", "complete", "single"], index=0, key="dist_link")

        if st.button("📊 Run Rigorous Distance Comparison", use_container_width=True):
            with st.spinner("Computing distance matrices and evaluating cluster validity..."):
                dist_res = ClusteringEngine.compare_distance_metrics(X_scaled, n_clusters=k_dist, linkage=link_dist)
                st.session_state["dist_res"] = dist_res

        dist_res = st.session_state.get("dist_res", None)
        if dist_res:
            st.markdown("#### Distance Metric Benchmark Results:")
            st.dataframe(dist_res["comparison_df"], use_container_width=True)

            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.markdown("##### Agglomerative Euclidean ($L_2$)")
                st.write(f"- **Silhouette:** `{dist_res['res_euclidean']['silhouette']:.4f}`")
                st.write(f"- **Davies-Bouldin:** `{dist_res['res_euclidean']['davies_bouldin']:.4f}`")
                st.write(f"- **Runtime:** `{dist_res['res_euclidean']['runtime_ms']} ms`")
            with d_col2:
                st.markdown("##### Agglomerative Manhattan ($L_1$)")
                st.write(f"- **Silhouette:** `{dist_res['res_manhattan']['silhouette']:.4f}`")
                st.write(f"- **Davies-Bouldin:** `{dist_res['res_manhattan']['davies_bouldin']:.4f}`")
                st.write(f"- **Runtime:** `{dist_res['res_manhattan']['runtime_ms']} ms`")

    # ==========================================
    # TAB 3: OPTIMIZATION CURVES
    # ==========================================
    with tab_curves:
        st.markdown("### Hyperparameter Optimization Curves ($K = 2$ to $10$)")
        st.markdown("""
        <div style="color: #94A3B8; font-size: 13px; margin-bottom: 12px;">
            Evaluating clustering performance across multiple granularities to identify optimal cluster separation.
        </div>
        """, unsafe_allow_html=True)

        if "elbow_curve_df" not in st.session_state:
            with st.spinner("Calculating WCSS, Silhouette, and Davies-Bouldin curves for K = 2 to 10..."):
                st.session_state["elbow_curve_df"] = ClusteringEngine.compute_elbow_curve(X_scaled, k_min=2, k_max=10)

        curve_df = st.session_state["elbow_curve_df"]

        # Display 3 curves
        fig_elbow = plot_elbow_curve(curve_df)
        st.plotly_chart(fig_elbow, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_sil = plot_silhouette_curve(curve_df)
            st.plotly_chart(fig_sil, use_container_width=True)
        with c2:
            fig_db = plot_davies_bouldin_curve(curve_df)
            st.plotly_chart(fig_db, use_container_width=True)

        st.markdown("#### Tabular Optimization Scores:")
        st.dataframe(curve_df, use_container_width=True)

    # ==========================================
    # TAB 4: PCA VISUALIZATION
    # ==========================================
    with tab_pca:
        st.markdown("### Principal Component Analysis (PCA) Latent Space Projection")
        
        pca_res = st.session_state.get("pca_results", None)
        active_res = st.session_state.get("active_lab_model", st.session_state.get("clustering_results", None))
        cluster_meta = st.session_state.get("cluster_metadata", {})

        if pca_res and active_res:
            st.markdown("""
            <div class="academic-callout">
                <strong>Methodological Clarification:</strong> PCA performs an orthogonal linear projection 
                maximizing captured variance. It is strictly utilized for visual inspection of cluster separability 
                in 2D/3D dimensions, <strong>not as a causal model</strong>.
            </div>
            """, unsafe_allow_html=True)

            pca_2d_tab, pca_3d_tab, pca_loadings_tab = st.tabs([
                "📈 2D PCA Cluster Map",
                "🌐 3D PCA Interactive Space",
                "📊 Component Loadings & Variance",
            ])

            with pca_2d_tab:
                fig_2d = plot_pca_2d(pca_res["coordinates"], active_res["labels"], cluster_meta)
                st.plotly_chart(fig_2d, use_container_width=True)

            with pca_3d_tab:
                fig_3d = plot_pca_3d(pca_res["coordinates"], active_res["labels"], cluster_meta)
                st.plotly_chart(fig_3d, use_container_width=True)

            with pca_loadings_tab:
                exp_vars = pca_res["explained_variance_ratio"]
                cum_vars = pca_res["cumulative_variance"]
                var_df = pd.DataFrame({
                    "Principal Component": [f"PC{i+1}" for i in range(len(exp_vars))],
                    "Explained Variance (%)": [f"{v*100:.2f}%" for v in exp_vars],
                    "Cumulative Variance (%)": [f"{c*100:.2f}%" for c in cum_vars],
                })
                st.markdown("##### Variance Explained:")
                st.dataframe(var_df, use_container_width=True)

                st.markdown("##### Eigenvector Feature Loadings Matrix:")
                st.dataframe(pca_res["loadings_df"], use_container_width=True)

    # ==========================================
    # TAB 5: MODEL BENCHMARK MATRIX
    # ==========================================
    with tab_benchmarks:
        st.markdown("### 📋 Historical Experiment Benchmark Table")
        exp_list = st.session_state.get("experiment_history", [])
        if len(exp_list) == 0:
            st.info("No experiment runs recorded yet. Fit models in the Algorithm Sandbox or Distance tabs to populate this benchmark.")
        else:
            bench_df = pd.DataFrame(exp_list)
            st.dataframe(bench_df, use_container_width=True)
            
            # Identify best model
            best_sil = bench_df.loc[bench_df["Silhouette Score"].idxmax()]
            best_db = bench_df.loc[bench_df["Davies-Bouldin Index"].idxmin()]
            
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.success(f"🏆 **Highest Silhouette Separation:** {best_sil['Algorithm']} ({best_sil['Settings']}) &mdash; `{best_sil['Silhouette Score']:.4f}`")
            with b_col2:
                st.success(f"🎯 **Lowest Davies-Bouldin Index:** {best_db['Algorithm']} ({best_db['Settings']}) &mdash; `{best_db['Davies-Bouldin Index']:.4f}`")


def _log_experiment(algo: str, settings: str, res: dict):
    """Logs an experiment run into session history."""
    entry = {
        "Algorithm": algo,
        "Settings": settings,
        "Clusters (K)": res["n_clusters"],
        "Silhouette Score": res["silhouette"],
        "Davies-Bouldin Index": res["davies_bouldin"],
        "Calinski-Harabasz": res["calinski_harabasz"],
        "Runtime (ms)": res["runtime_ms"],
    }
    st.session_state["experiment_history"].append(entry)
