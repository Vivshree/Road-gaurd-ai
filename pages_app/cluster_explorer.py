"""
ROADGUARD AI - Cluster Explorer & Semantic Profiling
Allows dynamic adjustment of K (2-10), recalculates cluster assignments, synthesizes descriptive
labels, and provides multi-attribute radar comparisons.
"""

import streamlit as st
import pandas as pd
from src.clustering import ClusteringEngine
from src.pca_engine import PCAEngine
from src.profiling import ClusterProfiler
from src.visualizer import plot_cluster_radar
from src.report_engine import ReportEngine
from src.styles import render_kpi

def render_cluster_explorer():
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 22px; font-weight: 800; color: #0F172A; margin-bottom: 4px;">
            Accident Cluster Explorer &amp; Semantic Profiler
        </h2>
        <div style="color: #64748B; font-size: 13.5px;">
            Unsupervised discovery of latent accident patterns. Adjust cluster granularity (K), 
            inspect data-driven profiles, and download cluster assignments.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.get("df_processed", None)
    X_scaled = st.session_state.get("X_scaled", None)

    if df is None or X_scaled is None:
        st.warning("No preprocessed dataset available. Please visit the Dataset Upload Center first.")
        return

    # K Selector & Re-run clustering
    col_k_sel, col_k_btn = st.columns([3, 1])
    with col_k_sel:
        current_k = st.session_state.get("cluster_k", 4)
        new_k = st.slider(
            "Select Number of Clusters (K)",
            min_value=2,
            max_value=10,
            value=current_k,
            step=1,
            key="slider_cluster_explorer_k",
        )

    # Re-run if K changed or user clicks Re-Cluster
    if new_k != current_k:
        with st.spinner(f"Re-running K-Means clustering with K = {new_k}..."):
            km_results = ClusteringEngine.run_kmeans(X_scaled, n_clusters=new_k, random_state=42)
            df["Cluster"] = km_results["labels"]
            summary_df, cluster_meta = ClusterProfiler.generate_cluster_profiles(df, cluster_col="Cluster")
            
            prep_summary = st.session_state.get("preprocessing_summary", {})
            features = prep_summary.get("features_used", [])
            pca_results = PCAEngine.fit_transform(X_scaled, feature_names=features)
            deviation_df = ClusterProfiler.compute_feature_deviations(df, feature_names=features, cluster_col="Cluster")

            st.session_state["df_processed"] = df
            st.session_state["cluster_k"] = new_k
            st.session_state["clustering_results"] = km_results
            st.session_state["pca_results"] = pca_results
            st.session_state["cluster_summary_df"] = summary_df
            st.session_state["cluster_metadata"] = cluster_meta
            st.session_state["feature_deviations_df"] = deviation_df
            st.rerun()

    cluster_meta = st.session_state.get("cluster_metadata", {})
    summary_df = st.session_state.get("cluster_summary_df", pd.DataFrame())

    # Academic Disclaimer Callout
    st.markdown("""
    <div class="academic-callout">
        <strong>Academic Guardrail &mdash; Data-Driven vs. Statutory Categorization:</strong>
        All cluster labels (e.g. <em>'Severe Night-Time Collision Corridor'</em>, <em>'Precipitation Skid Cluster'</em>) 
        are <strong>statistically derived descriptors</strong> synthesized from empirical mean measurements. 
        They do not represent legally recognized fault categories or official police classifications.
    </div>
    """, unsafe_allow_html=True)

    # 1. Cluster Summary Cards (Grid)
    st.markdown("### Discovered Cluster Archetypes")
    meta_cols = st.columns(len(cluster_meta))
    for idx, (c_id, meta) in enumerate(cluster_meta.items()):
        with meta_cols[idx]:
            card_html = f"""
            <div class="kpi-card" style="border-top: 3px solid {meta['color']};">
                <div style="font-weight: 800; font-size: 14px; color: {meta['color']}; margin-bottom: 2px;">
                    Cluster {c_id}
                </div>
                <div style="font-size: 12px; font-weight: 700; color: #0F172A; min-height: 32px;">
                    {meta['label']}
                </div>
                <div style="font-size: 20px; font-weight: 800; color: #0F172A; font-family: monospace; margin: 6px 0;">
                    {meta['size']:,} <span style="font-size: 11px; font-weight: 600; color: #64748B;">({meta['pct']}%)</span>
                </div>
                <div style="font-size: 11.5px; color: #334155; line-height: 1.5;">
                    &bull; Fatality: <strong>{meta['fatality_rate']}%</strong><br/>
                    &bull; Night-time: <strong>{meta['night_pct']}%</strong><br/>
                    &bull; Avg Speed: <strong>{meta['avg_speed']} km/h</strong><br/>
                    &bull; Road: <strong>{meta['top_road']}</strong>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 2. Multi-Attribute Radar Comparison
    c_left, c_right = st.columns([1.4, 1])
    with c_left:
        st.markdown("### 🕸️ Multi-Attribute Cluster Profiles (Normalized)")
        fig_radar = plot_cluster_radar(cluster_meta)
        st.plotly_chart(fig_radar, use_container_width=True)

    with c_right:
        st.markdown("### 📊 Cluster Membership Breakdown")
        size_df = summary_df[["Cluster ID", "Accident Count", "Share (%)", "Risk Classification"]]
        st.dataframe(size_df, use_container_width=True)

        km_res = st.session_state.get("clustering_results", {})
        st.markdown("##### Unsupervised Validation Scores:")
        st.write(f"- **Silhouette Score:** `{km_res.get('silhouette', 0.0):.4f}`")
        st.write(f"- **Davies-Bouldin Index:** `{km_res.get('davies_bouldin', 0.0):.4f}`")
        st.write(f"- **Inertia / WCSS:** `{km_res.get('inertia', 0.0):,.1f}`")

    # 3. Comprehensive Cluster Comparison Table
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### 📋 Empirical Cluster Comparison Matrix")
    st.dataframe(summary_df, use_container_width=True)

    # Download Button
    csv_data = ReportEngine.export_csv(df)
    st.download_button(
        label="📥 Download Complete Dataset with Cluster Assignments (CSV)",
        data=csv_data,
        file_name=f"roadguard_clustered_data_k{new_k}.csv",
        mime="text/csv",
    )
