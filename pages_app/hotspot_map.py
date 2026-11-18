"""
ROADGUARD AI - Accident Hotspot Geographical Map
Provides full-screen interactive spatial visualization, heatmaps, cluster-specific color rendering,
and multi-attribute contextual filters.
"""

import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
from src.map_engine import MapEngine

def render_hotspot_map():
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 22px; font-weight: 800; color: #FFFFFF; margin-bottom: 4px;">
            Geographical Accident Hotspot Intelligence Map
        </h2>
        <div style="color: #94A3B8; font-size: 13.5px;">
            Interactive spatial mapping displaying cluster-assigned accident coordinates, density heatmaps, and forensic popups.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.get("df_processed", None)
    if df is None or len(df) == 0:
        st.warning("No processed dataset found. Please visit the Dataset Upload Center to load data.")
        return

    # Check coordinate availability
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        st.error("""
        **Geographical Limitation:** The active dataset does not contain geographical coordinates (`Latitude` and `Longitude`).
        In accordance with scientific and academic integrity, **ROADGUARD AI strictly avoids inventing or hallucinating coordinates**.
        """)
        st.markdown("#### Tabular Location Breakdown (Alternative):")
        if "City" in df.columns:
            loc_summary = df.groupby("City").agg(
                Total_Accidents=("Accident_ID", "count"),
                Fatalities=("Number_of_Fatalities", "sum"),
                Injuries=("Number_of_Injuries", "sum"),
            ).reset_index().sort_values("Total_Accidents", ascending=False)
            st.dataframe(loc_summary, use_container_width=True)
        return

    # 1. Filters Bar
    with st.expander("⚙️ Spatial Filters &amp; Layer Controls", expanded=True):
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

        with m_col1:
            severities = sorted(df["Accident_Severity"].dropna().unique().tolist()) if "Accident_Severity" in df.columns else []
            sel_sev = st.multiselect("Severity", options=severities, default=severities)

        with m_col2:
            weather_opts = sorted(df["Weather"].dropna().unique().tolist()) if "Weather" in df.columns else []
            sel_weather = st.multiselect("Atmosphere / Weather", options=weather_opts, default=weather_opts)

        with m_col3:
            light_opts = sorted(df["Lighting_Condition"].dropna().unique().tolist()) if "Lighting_Condition" in df.columns else []
            sel_light = st.multiselect("Ambient Lighting", options=light_opts, default=light_opts)

        with m_col4:
            cluster_opts = sorted(df["Cluster"].dropna().unique().tolist()) if "Cluster" in df.columns else []
            cluster_display_opts = ["All Clusters"] + [f"Cluster {c}" for c in cluster_opts]
            sel_cluster_str = st.selectbox("Cluster Segmentation", options=cluster_display_opts, index=0)

        with m_col5:
            enable_heatmap = st.checkbox("Overlay Density Heatmap", value=True)
            marker_density = st.slider("Max Visible Markers", min_value=200, max_value=2500, value=1200, step=100)

    # Filter records
    filtered_df = df.copy()
    if sel_sev and "Accident_Severity" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Accident_Severity"].isin(sel_sev)]
    if sel_weather and "Weather" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Weather"].isin(sel_weather)]
    if sel_light and "Lighting_Condition" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Lighting_Condition"].isin(sel_light)]
    if sel_cluster_str != "All Clusters" and "Cluster" in filtered_df.columns:
        c_num = int(sel_cluster_str.split()[1])
        filtered_df = filtered_df[filtered_df["Cluster"] == c_num]

    if len(filtered_df) == 0:
        st.info("No accident points match the selected spatial criteria.")
        return

    # Render Map
    cluster_meta = st.session_state.get("cluster_metadata", None)
    folium_map, map_notice = MapEngine.create_accident_map(
        df=filtered_df,
        cluster_metadata=cluster_meta,
        enable_heatmap=enable_heatmap,
        max_markers=marker_density,
    )

    if map_notice:
        st.caption(f"ℹ️ {map_notice}")

    if folium_map:
        st_folium(folium_map, use_container_width=True, height=620)
    else:
        st.error("Failed to compile spatial visualization.")

    # Spatial Statistics Bar
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Spatial Subset Overview")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Points Displayed", f"{len(filtered_df):,}")
    s_col2.metric("Fatalities in Region", int(filtered_df["Number_of_Fatalities"].sum()) if "Number_of_Fatalities" in filtered_df.columns else 0)
    s_col3.metric("Injuries in Region", int(filtered_df["Number_of_Injuries"].sum()) if "Number_of_Injuries" in filtered_df.columns else 0)
    s_col4.metric("Avg Regional Speed", f"{filtered_df['Speed_Limit'].mean():.1f} km/h" if "Speed_Limit" in filtered_df.columns else "N/A")
