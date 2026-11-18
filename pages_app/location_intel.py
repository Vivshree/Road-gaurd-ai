"""
ROADGUARD AI - Location Intelligence & Decision Support
Provides hyper-local collision diagnostics, peer-corridor comparisons, and data-supported
safety intervention considerations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.styles import render_kpi
from src.visualizer import apply_dark_theme
from src.config import CLUSTER_COLORS

def render_location_intel():
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 22px; font-weight: 800; color: #0F172A; margin-bottom: 4px;">
            Location Intelligence &amp; Municipal Decision Support
        </h2>
        <div style="color: #64748B; font-size: 13.5px;">
            Deep-dive into specific city or corridor nodes. Compare local patterns against peer locations and evaluate data-supported safety measures.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.get("df_processed", None)
    if df is None or len(df) == 0:
        st.warning("No processed dataset found. Please visit the Dataset Upload Center to load data.")
        return

    # Select Location / City
    loc_col = "City" if "City" in df.columns else ("State" if "State" in df.columns else None)
    if not loc_col:
        st.info("No geographical jurisdiction column (City or State) available in dataset.")
        return

    available_locs = sorted(df[loc_col].dropna().unique().tolist())
    selected_loc = st.selectbox(
        f"Select Municipal Jurisdiction / Node ({loc_col})",
        options=available_locs,
        index=0,
    )

    loc_df = df[df[loc_col] == selected_loc]
    loc_total = len(loc_df)

    if loc_total == 0:
        st.info(f"No records available for {selected_loc}.")
        return

    cluster_meta = st.session_state.get("cluster_metadata", {})

    # Dominant cluster in this city
    if "Cluster" in loc_df.columns:
        dominant_cluster = int(loc_df["Cluster"].mode()[0])
        dom_meta = cluster_meta.get(dominant_cluster, {})
        dom_label = dom_meta.get("label", f"Cluster {dominant_cluster}")
    else:
        dominant_cluster = 0
        dom_label = "Unclustered"

    fatalities = int(loc_df["Number_of_Fatalities"].sum()) if "Number_of_Fatalities" in loc_df.columns else 0
    injuries = int(loc_df["Number_of_Injuries"].sum()) if "Number_of_Injuries" in loc_df.columns else 0
    avg_speed = round(float(loc_df["Speed_Limit"].mean()), 1) if "Speed_Limit" in loc_df.columns else 50.0
    night_pct = round(float(loc_df["Is_Night"].mean() * 100), 1) if "Is_Night" in loc_df.columns else 0.0
    weather_pct = round(float(loc_df["Is_Adverse_Weather"].mean() * 100), 1) if "Is_Adverse_Weather" in loc_df.columns else 0.0

    # KPI Strip
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(render_kpi("Local Collisions", f"{loc_total:,}", selected_loc, "cyan"), unsafe_allow_html=True)
    with k2:
        st.markdown(render_kpi("Dominant Cluster", f"Cluster {dominant_cluster}", dom_label[:20], "orange"), unsafe_allow_html=True)
    with k3:
        st.markdown(render_kpi("Fatalities", f"{fatalities:,}", f"{(fatalities/max(loc_total,1)*100):.1f}% Fatal", "red"), unsafe_allow_html=True)
    with k4:
        st.markdown(render_kpi("Injuries", f"{injuries:,}", "Casualties", "green"), unsafe_allow_html=True)
    with k5:
        st.markdown(render_kpi("Night Incidents", f"{night_pct}%", "20:00 - 06:00", "cyan"), unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 2. Local Visualizations
    col_v1, col_v2 = st.columns([1.2, 1])

    with col_v1:
        st.markdown(f"#### Cluster Composition in {selected_loc}")
        if "Cluster" in loc_df.columns:
            c_dist = loc_df["Cluster"].value_counts().reset_index()
            c_dist.columns = ["Cluster", "Count"]
            c_dist["Cluster_Name"] = c_dist["Cluster"].apply(
                lambda c: f"Cluster {c}: {cluster_meta.get(c, {}).get('label', '')[:22]}"
            )
            fig_pie = px.pie(
                c_dist,
                names="Cluster_Name",
                values="Count",
                hole=0.5,
                color_discrete_sequence=CLUSTER_COLORS,
                title=f"Accident Share by Cluster in {selected_loc}",
            )
            st.plotly_chart(apply_dark_theme(fig_pie), use_container_width=True)
        else:
            st.info("Clustering not fitted.")

    with col_v2:
        st.markdown(f"#### Road Type Risk Breakdown ({selected_loc})")
        if "Road_Type" in loc_df.columns:
            road_dist = loc_df.groupby("Road_Type").agg(
                Accidents=("Accident_ID", "count"),
                Fatalities=("Number_of_Fatalities", "sum"),
            ).reset_index().sort_values("Accidents", ascending=False)
            fig_bar = px.bar(
                road_dist,
                x="Road_Type",
                y="Accidents",
                color="Fatalities",
                color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                title=f"Collisions &amp; Fatalities by Roadway",
            )
            st.plotly_chart(apply_dark_theme(fig_bar), use_container_width=True)
        else:
            st.info("Road_Type not available.")

    # 3. Peer Comparison across jurisdictions
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown(f"#### Peer Comparison: Other Nodes in Cluster {dominant_cluster} ({dom_label})")
    
    if "Cluster" in df.columns:
        peer_df = df[df["Cluster"] == dominant_cluster]
        peer_summary = peer_df.groupby(loc_col).agg(
            Cluster_Accidents=("Accident_ID", "count"),
            Fatalities=("Number_of_Fatalities", "sum"),
            Injuries=("Number_of_Injuries", "sum"),
            Avg_Speed=("Speed_Limit", "mean"),
        ).reset_index().sort_values("Cluster_Accidents", ascending=False)
        peer_summary["Avg_Speed"] = peer_summary["Avg_Speed"].round(1)
        st.dataframe(peer_summary, use_container_width=True)

    # 4. Data-Supported Safety Intervention Hypotheses
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🛡️ Data-Supported Road Safety Considerations")

    st.markdown("""
    <div class="academic-callout">
        <strong>Academic Guardrail &mdash; Exploratory Decision-Support (Non-Causal):</strong><br/>
        The safety considerations below are generated based on empirical statistical traits observed in this location's data. 
        <strong>The unsupervised model proves correlation, not causation.</strong> These recommendations should be treated 
        as prioritized hypotheses for civil engineers, traffic police, and municipal authorities.
    </div>
    """, unsafe_allow_html=True)

    rec_cols = st.columns(3)

    with rec_cols[0]:
        st.markdown("""
        <div class="dashboard-panel" style="min-height: 180px;">
            <div style="color: #0284C7; font-weight: 700; font-size: 14px; margin-bottom: 6px;">
                💡 Illumination &amp; Nocturnal Interventions
            </div>
            <div style="font-size: 12.5px; color: #334155; line-height: 1.5;">
        """, unsafe_allow_html=True)
        if night_pct >= 40.0:
            st.write(f"- Night-time accidents account for **{night_pct}%** of local occurrences.")
            st.write("- Priority: Audit streetlighting lux levels along high-incident corridors.")
            st.write("- Install high-intensity solar retro-reflective cat-eyes and roadside delineators.")
        else:
            st.write(f"- Night-time incidents represent a moderate proportion (**{night_pct}%**).")
            st.write("- Maintain regular inspection of junction lighting timers and signal visibility.")
        st.markdown("</div></div>", unsafe_allow_html=True)

    with rec_cols[1]:
        st.markdown("""
        <div class="dashboard-panel" style="min-height: 180px;">
            <div style="color: #D97706; font-weight: 700; font-size: 14px; margin-bottom: 6px;">
                🌧️ Pavement &amp; Weather Resilience
            </div>
            <div style="font-size: 12.5px; color: #334155; line-height: 1.5;">
        """, unsafe_allow_html=True)
        if weather_pct >= 25.0:
            st.write(f"- Adverse weather correlates with **{weather_pct}%** of collisions in this node.")
            st.write("- Priority: High-friction anti-skid surface dressing at curvature approaches.")
            st.write("- Improve stormwater drainage cross-slopes to mitigate dangerous aquaplaning.")
        else:
            st.write(f"- Low weather sensitivity observed (**{weather_pct}%** wet/foggy incidents).")
            st.write("- Focus on dry-surface pavement rutting, pothole repair, and standard lane striping.")
        st.markdown("</div></div>", unsafe_allow_html=True)

    with rec_cols[2]:
        st.markdown("""
        <div class="dashboard-panel" style="min-height: 180px;">
            <div style="color: #DC2626; font-weight: 700; font-size: 14px; margin-bottom: 6px;">
                ⚡ Speed Calming &amp; Enforcement
            </div>
            <div style="font-size: 12.5px; color: #334155; line-height: 1.5;">
        """, unsafe_allow_html=True)
        if avg_speed >= 70.0:
            st.write(f"- Average corridor speed limit is elevated (**{avg_speed} km/h**).")
            st.write("- Priority: Deploy automated sectional speed detection cameras (ANPR).")
            st.write("- Install progressive rumble strips and variable message signs (VMS) at merges.")
        else:
            st.write(f"- Average speed is urban-paced (**{avg_speed} km/h**).")
            st.write("- Deploy pedestrian refuge islands, raised crosswalks, and junction signal retiming.")
        st.markdown("</div></div>", unsafe_allow_html=True)
