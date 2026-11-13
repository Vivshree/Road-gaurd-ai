"""
ROADGUARD AI - Road Accident Intelligence & Hotspot Segmentation System
Main Streamlit Application Entrypoint & Session Orchestrator.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Configure Streamlit page layout
st.set_page_config(
    page_title="ROADGUARD AI — Road Safety Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import internal modules
from src.config import APP_NAME, APP_TAGLINE, APP_SUBTITLE, PAGES
from src.styles import inject_custom_styles, render_header, render_footer
from src.data_loader import load_sample_dataset, auto_map_columns
from pages_app.upload_center import _run_pipeline, render_upload_center
from pages_app.executive_overview import render_executive_overview
from pages_app.hotspot_map import render_hotspot_map
from pages_app.cluster_explorer import render_cluster_explorer
from pages_app.ml_lab import render_ml_lab
from pages_app.pattern_analysis import render_pattern_analysis
from pages_app.location_intel import render_location_intel
from pages_app.report_generator import render_report_generator

# Inject custom CSS
inject_custom_styles()

def initialize_session_state():
    """Initializes default application session state on startup with benchmark dataset."""
    if "df_raw" not in st.session_state:
        # Load prepackaged academic benchmark dataset
        df_sample = load_sample_dataset()
        st.session_state["df_raw"] = df_sample
        st.session_state["dataset_name"] = "Synthetic Academic Benchmark"
        st.session_state["dataset_source"] = "Synthetic Multi-Corridor Road Accident Generator"
        st.session_state["column_mapping"] = auto_map_columns(df_sample.columns.tolist())
        st.session_state["cluster_k"] = 4
        
        # Execute default pipeline so app starts fully operational
        _run_pipeline(df_sample, k=4)

initialize_session_state()

# -------------------------------------------------------------
# SIDEBAR NAVIGATION & SYSTEM STATUS
# -------------------------------------------------------------
with st.sidebar:
    # Sidebar Logo & Title
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 14px;">
        <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #0284C7 0%, #22D3EE 100%); 
                    border-radius: 10px; display: flex; align-items: center; justify-content: center; 
                    box-shadow: 0 4px 14px rgba(34, 211, 238, 0.35);">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="M12 8v4"/>
                <path d="M12 16h.01"/>
            </svg>
        </div>
        <div>
            <div style="font-weight: 800; font-size: 18px; color: #FFFFFF; letter-spacing: -0.3px;">ROADGUARD AI</div>
            <div style="font-size: 11px; color: #22D3EE; font-weight: 500;">Safety Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Page Navigation Selector
    st.markdown("#### Navigation")
    current_page = st.radio(
        "Select Dashboard View",
        options=PAGES,
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Dataset Status Panel
    st.markdown("#### System & Dataset Status")
    ds_name = st.session_state.get("dataset_name", "Academic Benchmark")
    total_recs = len(st.session_state.get("df_processed", []))
    k_val = st.session_state.get("cluster_k", 4)
    sil_score = st.session_state.get("clustering_results", {}).get("silhouette", 0.0)

    st.markdown(f"""
    <div style="background: #131E30; border: 1px solid #1E293B; border-radius: 8px; padding: 12px; font-size: 12px; margin-bottom: 12px;">
        <div style="color: #94A3B8; margin-bottom: 4px;">ACTIVE DATASET</div>
        <div style="font-weight: 700; color: #FFFFFF; margin-bottom: 6px; word-break: break-word;">{ds_name}</div>
        <div style="color: #CBD5E1; line-height: 1.5;">
            &bull; <strong>Records:</strong> {total_recs:,}<br/>
            &bull; <strong>Clusters (K):</strong> {k_val}<br/>
            &bull; <strong>Silhouette:</strong> {sil_score:.4f}<br/>
            &bull; <strong>Status:</strong> <span style="color: #10B981; font-weight: 600;">Operational</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Reset to Default Benchmark", use_container_width=True):
        df_sample = load_sample_dataset()
        st.session_state["df_raw"] = df_sample
        st.session_state["dataset_name"] = "Synthetic Academic Benchmark"
        st.session_state["dataset_source"] = "Synthetic Multi-Corridor Road Accident Generator"
        st.session_state["column_mapping"] = auto_map_columns(df_sample.columns.tolist())
        _run_pipeline(df_sample, k=4)
        st.success("Reset to benchmark dataset!")
        st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 11px; color: #64748B; text-align: center; border-top: 1px solid #1E293B; padding-top: 10px;">
        Academic Project &bull; Unsupervised ML<br/>
        Decision Support System
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# MAIN CONTENT AREA
# -------------------------------------------------------------
render_header()

if current_page == "Dataset Upload Center":
    render_upload_center()
elif current_page == "Executive Overview":
    render_executive_overview()
elif current_page == "Accident Hotspot Map":
    render_hotspot_map()
elif current_page == "Cluster Explorer":
    render_cluster_explorer()
elif current_page == "Machine Learning Lab":
    render_ml_lab()
elif current_page == "Pattern Analysis":
    render_pattern_analysis()
elif current_page == "Location Intelligence":
    render_location_intel()
elif current_page == "Report Generator":
    render_report_generator()

render_footer()
