"""
ROADGUARD AI - Executive Overview Dashboard
Presents high-level traffic intelligence KPIs, temporal trends, atmospheric patterns,
top high-risk corridors, and dynamic multi-attribute filtering.
"""

import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
from src.styles import render_kpi
from src.visualizer import (
    plot_executive_trends, plot_severity_distribution,
    plot_accidents_by_hour, plot_weather_lighting
)
from src.map_engine import MapEngine

def render_executive_overview():
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 22px; font-weight: 800; color: #0F172A; margin-bottom: 4px;">
            Executive Traffic Intelligence Overview
        </h2>
        <div style="color: #64748B; font-size: 13.5px;">
            Strategic command-center dashboard tracking network-wide accident frequencies, severity ratios, and hotspot distributions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.get("df_processed", None)
    if df is None or len(df) == 0:
        st.warning("No processed dataset found. Please visit the Dataset Upload Center to load data.")
        return

    # 1. Interactive Dynamic Filters
    with st.expander("🔍 Strategic Filter Controls", expanded=False):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            states_available = sorted(df["State"].dropna().unique().tolist()) if "State" in df.columns else []
            selected_states = st.multiselect("State / Region", options=states_available, default=states_available)

        with f_col2:
            cities_available = sorted(df["City"].dropna().unique().tolist()) if "City" in df.columns else []
            selected_cities = st.multiselect("City / District", options=cities_available, default=cities_available)

        with f_col3:
            severities_available = sorted(df["Accident_Severity"].dropna().unique().tolist()) if "Accident_Severity" in df.columns else []
            selected_severities = st.multiselect("Severity", options=severities_available, default=severities_available)

        with f_col4:
            roads_available = sorted(df["Road_Type"].dropna().unique().tolist()) if "Road_Type" in df.columns else []
            selected_roads = st.multiselect("Road Type", options=roads_available, default=roads_available)

    # Apply filters
    filtered_df = df.copy()
    if selected_states and "State" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["State"].isin(selected_states)]
    if selected_cities and "City" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["City"].isin(selected_cities)]
    if selected_severities and "Accident_Severity" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Accident_Severity"].isin(selected_severities)]
    if selected_roads and "Road_Type" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Road_Type"].isin(selected_roads)]

    if len(filtered_df) == 0:
        st.info("No accident records match the selected filters. Please expand your selection.")
        return

    # 2. Executive KPI Cards
    total_accidents = len(filtered_df)
    fatal_accidents = int((filtered_df["Number_of_Fatalities"] > 0).sum()) if "Number_of_Fatalities" in filtered_df.columns else 0
    total_injuries = int(filtered_df["Number_of_Injuries"].sum()) if "Number_of_Injuries" in filtered_df.columns else 0
    locations_count = int(filtered_df["City"].nunique()) if "City" in filtered_df.columns else 1
    identified_clusters = int(filtered_df["Cluster"].nunique()) if "Cluster" in filtered_df.columns else 1
    avg_severity = round(float(filtered_df["Severity_Numeric"].mean()), 2) if "Severity_Numeric" in filtered_df.columns else 1.5

    fatality_rate_str = f"{(fatal_accidents / max(total_accidents, 1) * 100):.1f}% Fatal"
    
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    with kpi1:
        st.markdown(render_kpi("Total Collisions", f"{total_accidents:,}", "Network Total", "cyan"), unsafe_allow_html=True)
    with kpi2:
        st.markdown(render_kpi("Fatal Incidents", f"{fatal_accidents:,}", fatality_rate_str, "red"), unsafe_allow_html=True)
    with kpi3:
        st.markdown(render_kpi("Injuries Sustained", f"{total_injuries:,}", "Casualties", "orange"), unsafe_allow_html=True)
    with kpi4:
        st.markdown(render_kpi("Cities / Nodes", str(locations_count), "Analyzed", "green"), unsafe_allow_html=True)
    with kpi5:
        st.markdown(render_kpi("Clusters Discovered", str(identified_clusters), "K-Means", "cyan"), unsafe_allow_html=True)
    with kpi6:
        st.markdown(render_kpi("Average Severity", f"{avg_severity}/3.0", "Ordinal Score", "orange"), unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 3. Interactive Charts (2x2 Grid)
    c_row1_left, c_row1_right = st.columns([1.5, 1])
    with c_row1_left:
        fig_trend = plot_executive_trends(filtered_df)
        st.plotly_chart(fig_trend, use_container_width=True)
    with c_row1_right:
        fig_sev = plot_severity_distribution(filtered_df)
        st.plotly_chart(fig_sev, use_container_width=True)

    c_row2_left, c_row2_right = st.columns([1, 1])
    with c_row2_left:
        fig_hour = plot_accidents_by_hour(filtered_df)
        st.plotly_chart(fig_hour, use_container_width=True)
    with c_row2_right:
        fig_env = plot_weather_lighting(filtered_df)
        st.plotly_chart(fig_env, use_container_width=True)

    # 4. Geographic Hotspot Map (Embedded Overview)
    st.markdown("### 🗺️ Geographic Hotspot Distribution")
    cluster_meta = st.session_state.get("cluster_metadata", None)
    folium_map, map_msg = MapEngine.create_accident_map(
        filtered_df, cluster_metadata=cluster_meta, enable_heatmap=True, max_markers=800
    )

    if map_msg:
        st.info(map_msg)

    if folium_map:
        st_folium(folium_map, use_container_width=True, height=460)
    else:
        st.info("Map disabled: Coordinates unavailable in the current dataset.")

    # 5. Top Accident-Prone Corridors Table
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🚨 Top High-Risk Corridors &amp; Locations")

    group_cols = []
    if "City" in filtered_df.columns:
        group_cols.append("City")
    if "Road_Type" in filtered_df.columns:
        group_cols.append("Road_Type")

    if group_cols:
        top_locations = filtered_df.groupby(group_cols).agg(
            Total_Accidents=("Accident_ID", "count"),
            Fatalities=("Number_of_Fatalities", "sum"),
            Injuries=("Number_of_Injuries", "sum"),
            Avg_Severity=("Severity_Numeric", "mean"),
        ).reset_index().sort_values("Total_Accidents", ascending=False).head(10)

        top_locations["Avg_Severity"] = top_locations["Avg_Severity"].round(2)
        st.dataframe(top_locations, use_container_width=True)
    else:
        st.dataframe(filtered_df.head(10), use_container_width=True)
