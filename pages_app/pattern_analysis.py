"""
ROADGUARD AI - Accident Pattern Analysis
Conducts multi-dimensional exploratory pattern analysis across temporal, atmospheric,
and infrastructural variables, along with standardized feature deviation profiling.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.visualizer import (
    plot_hour_weekday_heatmap, plot_feature_deviations, apply_dark_theme
)
from src.config import CLUSTER_COLORS, COLOR_PANEL

def render_pattern_analysis():
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 22px; font-weight: 800; color: #FFFFFF; margin-bottom: 4px;">
            Accident Pattern &amp; Feature Contribution Analysis
        </h2>
        <div style="color: #94A3B8; font-size: 13.5px;">
            Examine temporal concentrations, roadway infrastructure interactions, and descriptive feature deviation profiles.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.get("df_processed", None)
    if df is None or len(df) == 0:
        st.warning("No processed dataset found. Please visit the Dataset Upload Center to load data.")
        return

    # Section 1: Temporal Patterns
    st.markdown("### 1. Temporal Concentration Matrix")
    fig_heatmap = plot_hour_weekday_heatmap(df)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Section 2: Atmospheric & Infrastructure Interactions
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### 2. Environmental &amp; Infrastructure Interactions")

    col_env1, col_env2 = st.columns(2)
    with col_env1:
        if "Road_Surface" in df.columns and "Road_Type" in df.columns:
            surf_road = df.groupby(["Road_Type", "Road_Surface"]).size().reset_index(name="Accident_Count")
            fig_surf = px.bar(
                surf_road,
                x="Road_Type",
                y="Accident_Count",
                color="Road_Surface",
                title="Accidents by Road Classification &amp; Surface Condition",
                barmode="stack",
                color_discrete_sequence=CLUSTER_COLORS,
            )
            st.plotly_chart(apply_dark_theme(fig_surf), use_container_width=True)
        else:
            st.info("Road_Surface or Road_Type feature missing.")

    with col_env2:
        if "Lighting_Condition" in df.columns and "Accident_Severity" in df.columns:
            light_sev = df.groupby(["Lighting_Condition", "Accident_Severity"]).size().reset_index(name="Accident_Count")
            fig_light = px.bar(
                light_sev,
                x="Lighting_Condition",
                y="Accident_Count",
                color="Accident_Severity",
                title="Collision Severity by Ambient Lighting Environment",
                barmode="group",
                color_discrete_map={"Fatal": "#EF4444", "Serious": "#F59E0B", "Slight": "#10B981"},
            )
            st.plotly_chart(apply_dark_theme(fig_light), use_container_width=True)
        else:
            st.info("Lighting_Condition or Accident_Severity missing.")

    # Section 3: Feature Contribution / Z-Score Deviation Profiling
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### 3. Cluster Feature Contribution &amp; Deviation Profiling")

    st.markdown("""
    <div class="academic-callout">
        <strong>Academic Guardrail &mdash; Descriptive Profiling vs. Causal Attribution:</strong><br/>
        The standardized Z-scores below display the distance (in standard deviations) of each cluster centroid 
        from the overall dataset mean. They describe <em>how</em> a cluster differs from normal conditions. 
        <strong>They are strictly descriptive characteristics and must NOT be called causal feature importance values.</strong>
    </div>
    """, unsafe_allow_html=True)

    deviation_df = st.session_state.get("feature_deviations_df", None)
    cluster_meta = st.session_state.get("cluster_metadata", {})

    if deviation_df is not None and cluster_meta:
        c_ids = sorted(cluster_meta.keys())
        cluster_choices = {f"Cluster {cid}: {cluster_meta[cid]['label']}": cid for cid in c_ids}
        sel_choice = st.selectbox(
            "Select Cluster to Inspect Z-Score Feature Deviations",
            options=list(cluster_choices.keys()),
            index=0,
        )
        selected_cid = cluster_choices[sel_choice]
        selected_label = cluster_meta[selected_cid]["label"]

        fig_dev = plot_feature_deviations(deviation_df, selected_cid, selected_label)
        st.plotly_chart(fig_dev, use_container_width=True)

        st.markdown(f"#### Summary Statistics for Cluster {selected_cid}:")
        sub_dev = deviation_df[deviation_df["Cluster_ID"] == selected_cid][
            ["Feature", "Cluster_Mean", "Global_Mean", "Z_Score_Deviation"]
        ].sort_values("Z_Score_Deviation", ascending=False)
        st.dataframe(sub_dev, use_container_width=True)
    else:
        st.info("Fit a clustering model in the Machine Learning Lab to generate feature deviation profiles.")
