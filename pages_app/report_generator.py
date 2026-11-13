"""
ROADGUARD AI - Academic Report Generator & Export Center
Generates publication-quality PDF audit reports, provides CSV exports, and outlines academic viva defense notes.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from src.report_engine import ReportEngine

def render_report_generator():
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 22px; font-weight: 800; color: #FFFFFF; margin-bottom: 4px;">
            Academic Report Generator &amp; Data Export Center
        </h2>
        <div style="color: #94A3B8; font-size: 13.5px;">
            Compile presentation-ready PDF academic research reports and export processed collision data with cluster assignments.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.get("df_processed", None)
    if df is None or len(df) == 0:
        st.warning("No processed dataset found. Please visit the Dataset Upload Center to load data.")
        return

    dataset_name = st.session_state.get("dataset_name", "Academic Demonstration Dataset")
    km_res = st.session_state.get("clustering_results", {})
    summary_df = st.session_state.get("cluster_summary_df", pd.DataFrame())
    pca_res = st.session_state.get("pca_results", {})
    prep_summary = st.session_state.get("preprocessing_summary", {})

    col_rep_left, col_rep_right = st.columns([1.4, 1])

    with col_rep_left:
        st.markdown("### 📄 Publication PDF Research Report")
        st.markdown("""
        Generate an official multi-page PDF document detailing the experimental pipeline, mathematical hyperparameters, 
        unsupervised validation scores, empirical cluster profiles, and academic limitations.
        """)

        st.markdown(f"""
        <div class="dashboard-panel" style="padding: 16px 20px;">
            <div style="font-size: 14px; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
                Report Document Metadata
            </div>
            <div style="font-size: 12px; color: #CBD5E1; line-height: 1.6;">
                &bull; <strong>Title:</strong> ROADGUARD AI &mdash; Research &amp; Decision-Support Audit<br/>
                &bull; <strong>Dataset:</strong> {dataset_name} ({len(df):,} records)<br/>
                &bull; <strong>Algorithm:</strong> {km_res.get('algorithm', 'K-Means')} (K = {km_res.get('n_clusters', 4)})<br/>
                &bull; <strong>Validation Scores:</strong> Silhouette: {km_res.get('silhouette', 0.0):.4f} | DB Index: {km_res.get('davies_bouldin', 0.0):.4f}<br/>
                &bull; <strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔨 Compile Official PDF Academic Report", use_container_width=True):
            with st.spinner("Compiling academic document using ReportLab..."):
                try:
                    pdf_bytes = ReportEngine.generate_pdf_report(
                        dataset_name=dataset_name,
                        total_records=len(df),
                        clustering_results=km_res,
                        cluster_summary_df=summary_df,
                        pca_results=pca_res,
                        preprocessing_summary=prep_summary,
                    )
                    st.session_state["generated_pdf"] = pdf_bytes
                    st.success("PDF Report generated successfully!")
                except Exception as e:
                    st.error(f"PDF generation failed: {str(e)}")

        pdf_data = st.session_state.get("generated_pdf", None)
        if pdf_data:
            st.download_button(
                label="📥 Download Generated Academic PDF Report",
                data=pdf_data,
                file_name=f"ROADGUARD_AI_Research_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with col_rep_right:
        st.markdown("### 💾 CSV Data &amp; Benchmark Exports")
        st.markdown("Download raw and derived datasets for offline external analysis:")

        # 1. Processed CSV
        csv_df = ReportEngine.export_csv(df)
        st.download_button(
            label="📥 Download Processed Dataset + Clusters (CSV)",
            data=csv_df,
            file_name=f"roadguard_clustered_dataset_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # 2. Cluster Profiles CSV
        if not summary_df.empty:
            csv_summary = ReportEngine.export_csv(summary_df)
            st.download_button(
                label="📥 Download Cluster Profiles Matrix (CSV)",
                data=csv_summary,
                file_name="roadguard_cluster_profiles.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # 3. Model Benchmark CSV
        exp_history = st.session_state.get("experiment_history", [])
        if exp_history:
            csv_exp = ReportEngine.export_csv(pd.DataFrame(exp_history))
            st.download_button(
                label="📥 Download Model Benchmark Comparison (CSV)",
                data=csv_exp,
                file_name="roadguard_model_benchmarks.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # Academic Viva Defense Section
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🎓 Academic Project &amp; Viva Defense Brief")

    viva_col1, viva_col2 = st.columns(2)
    with viva_col1:
        st.markdown("""
        <div class="dashboard-panel">
            <div style="color: #22D3EE; font-weight: 700; font-size: 14px; margin-bottom: 6px;">
                Q1: Why Unsupervised Learning instead of Supervised Prediction?
            </div>
            <div style="font-size: 12.5px; color: #CBD5E1; line-height: 1.5;">
                In road safety intelligence, true causal ground-truth labels do not exist for future accidents. 
                Supervised models can predict historical outcomes but cannot discover novel, multi-variable collision 
                archetypes across unclassified spatial corridors. Unsupervised clustering partitions the multi-dimensional 
                space objectively to uncover latent risk structures.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="dashboard-panel">
            <div style="color: #F59E0B; font-weight: 700; font-size: 14px; margin-bottom: 6px;">
                Q2: Why does standard K-Means require Euclidean distance?
            </div>
            <div style="font-size: 12.5px; color: #CBD5E1; line-height: 1.5;">
                Standard Lloyd's K-Means minimizes squared Euclidean distance (L2 norm). In L2 space, the cluster 
                centroid that minimizes within-cluster variance is analytically proven to be the arithmetic mean. 
                Using Manhattan distance (L1 norm) requires the geometric median (as in K-Medians / PAM), otherwise 
                the centroid update is mathematically sub-optimal.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with viva_col2:
        st.markdown("""
        <div class="dashboard-panel">
            <div style="color: #10B981; font-weight: 700; font-size: 14px; margin-bottom: 6px;">
                Q3: How are Silhouette and Davies-Bouldin Scores interpreted?
            </div>
            <div style="font-size: 12.5px; color: #CBD5E1; line-height: 1.5;">
                <strong>Silhouette Score [-1 to +1]:</strong> Computes $(b - a) / \\max(a, b)$, where $a$ is mean intra-cluster distance and $b$ is nearest-cluster distance. Higher values indicate dense, well-separated clusters.<br/>
                <strong>Davies-Bouldin Index:</strong> Ratio of within-cluster scatter to between-cluster separation. Lower values signify superior partition compactness.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="dashboard-panel">
            <div style="color: #EF4444; font-weight: 700; font-size: 14px; margin-bottom: 6px;">
                Q4: What are the primary academic limitations of this system?
            </div>
            <div style="font-size: 12.5px; color: #CBD5E1; line-height: 1.5;">
                <strong>Reporting &amp; Geographical Bias:</strong> Minor accidents are heavily underreported compared to fatal incidents. 
                Clustering assumes spherical geometries in K-Means, which may fragment curved roadways. Most importantly, 
                cluster correlations provide decision-support hypotheses, <strong>not causal proof</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)
