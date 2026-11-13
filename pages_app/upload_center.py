"""
ROADGUARD AI - Dedicated Dataset Upload Center
Provides real-world file ingestion (CSV/Excel), automatic intelligent column mapping,
data quality inspection, preprocessing controls, and unsupervised ML execution.
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.config import CANONICAL_COLUMNS, DEFAULT_CLUSTERING_FEATURES
from src.data_loader import (
    read_uploaded_file, auto_map_columns, apply_column_mapping,
    inspect_dataset_quality, load_sample_dataset
)
from src.preprocessor import AccidentPreprocessor
from src.clustering import ClusteringEngine
from src.pca_engine import PCAEngine
from src.profiling import ClusterProfiler
from src.styles import render_kpi

def render_upload_center():
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 22px; font-weight: 800; color: #FFFFFF; margin-bottom: 4px;">
            Dataset Upload Center &amp; Ingestion Pipeline
        </h2>
        <div style="color: #94A3B8; font-size: 13.5px;">
            Upload real-world collision CSV or Excel records, map columns to canonical safety schemas, 
            configure pre-processing, and execute unsupervised clustering.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset Status Info Card
    current_name = st.session_state.get("dataset_name", "Synthetic Academic Benchmark")
    total_records = len(st.session_state.get("df_raw", pd.DataFrame()))
    has_clusters = "Cluster" in st.session_state.get("df_processed", pd.DataFrame()).columns

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.markdown(render_kpi("Active Dataset", current_name, "In Memory", "cyan"), unsafe_allow_html=True)
    with col_stat2:
        st.markdown(render_kpi("Total Records", f"{total_records:,}", "Ingested", "green"), unsafe_allow_html=True)
    with col_stat3:
        st.markdown(render_kpi("Preprocessing", "Ready" if has_clusters else "Pending", "Transformed", "orange"), unsafe_allow_html=True)
    with col_stat4:
        st.markdown(render_kpi("Clustering State", "Fitted" if has_clusters else "Awaiting", "K-Means", "cyan"), unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 1. Drag and Drop File Uploader
    with st.container():
        st.markdown("""
        <div class="upload-box">
            <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="#22D3EE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 8px;">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <div style="font-weight: 700; font-size: 16px; color: #FFFFFF; margin-bottom: 4px;">
                Upload Real-World Collision Dataset
            </div>
            <div style="color: #94A3B8; font-size: 13px;">
                Drag &amp; drop .CSV or .XLSX / .XLS files (up to 200MB)
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Select or drop accident dataset file",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
            key="real_file_uploader",
        )

        col_up_action1, col_up_action2 = st.columns([1, 1])
        with col_up_action1:
            if st.button("🔄 Reset to Default Academic Benchmark Dataset", use_container_width=True):
                with st.spinner("Reloading default benchmark records..."):
                    df_sample = load_sample_dataset()
                    st.session_state["df_raw"] = df_sample
                    st.session_state["dataset_name"] = "Synthetic Academic Benchmark"
                    st.session_state["dataset_source"] = "Synthetic Multi-Corridor Road Accident Generator"
                    st.session_state["column_mapping"] = auto_map_columns(df_sample.columns.tolist())
                    _run_pipeline(df_sample, k=st.session_state.get("cluster_k", 4))
                    st.success("Successfully reset to benchmark dataset!")
                    st.rerun()

    # Handle New Upload
    if uploaded_file is not None:
        filename = uploaded_file.name
        # Only parse if file changed
        if st.session_state.get("last_uploaded_file") != filename:
            with st.spinner(f"Parsing uploaded file '{filename}'..."):
                raw_df, err = read_uploaded_file(uploaded_file, filename)
                if err:
                    st.error(err)
                elif raw_df is not None:
                    st.session_state["df_raw"] = raw_df
                    st.session_state["dataset_name"] = filename
                    st.session_state["dataset_source"] = f"User Upload: {filename}"
                    st.session_state["last_uploaded_file"] = filename
                    st.session_state["column_mapping"] = auto_map_columns(raw_df.columns.tolist())
                    st.success(f"Parsed {len(raw_df):,} rows and {len(raw_df.columns)} columns successfully!")

    df_current_raw = st.session_state.get("df_raw", None)
    if df_current_raw is None or len(df_current_raw) == 0:
        st.info("No dataset is currently loaded. Click 'Reset to Default Academic Benchmark Dataset' or upload a file.")
        return

    # 2. Data Preview & Quality Inspection
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("### 2. Dataset Quality Audit &amp; Preview")
    quality = inspect_dataset_quality(df_current_raw)

    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    q_col1.metric("Total Records", f"{quality['total_rows']:,}")
    q_col2.metric("Total Columns", quality["total_cols"])
    q_col3.metric("Completeness", f"{quality['completeness_pct']}%")
    q_col4.metric("Health Index", f"{quality['health_score']}/100")

    tab_preview, tab_col_summary = st.tabs(["📄 Raw Data Preview (First 15 Rows)", "📊 Column Type & Missing Value Audit"])
    with tab_preview:
        st.dataframe(df_current_raw.head(15), use_container_width=True)
    with tab_col_summary:
        st.dataframe(quality["column_summary_df"], use_container_width=True)

    # 3. Intelligent Column Mapping
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### 3. Intelligent Schema Column Mapping")
    st.markdown("""
    <div class="academic-callout">
        <strong>Automatic Schema Alignment:</strong> ROADGUARD AI automatically matched raw dataset headers to canonical accident attributes. 
        Adjust any dropdown below if a column was misidentified or unmapped. Columns left as <em>(Not Present)</em> will be gracefully handled with fallback defaults.
    </div>
    """, unsafe_allow_html=True)

    raw_cols_options = ["(Not Present)"] + list(df_current_raw.columns)
    current_mapping = st.session_state.get("column_mapping", auto_map_columns(df_current_raw.columns.tolist()))

    map_cols = st.columns(3)
    user_mapping = {}

    for i, (canonical_name, meta) in enumerate(CANONICAL_COLUMNS.items()):
        col_idx = i % 3
        with map_cols[col_idx]:
            default_raw = current_mapping.get(canonical_name)
            default_index = 0
            if default_raw and default_raw in raw_cols_options:
                default_index = raw_cols_options.index(default_raw)

            req_marker = " ⚠️ (Core)" if meta["required"] else ""
            selected_raw = st.selectbox(
                f"{canonical_name}{req_marker}",
                options=raw_cols_options,
                index=default_index,
                key=f"map_sel_{canonical_name}",
                help=meta["desc"],
            )
            user_mapping[canonical_name] = None if selected_raw == "(Not Present)" else selected_raw

    st.session_state["column_mapping"] = user_mapping

    # 4. Preprocessing & Feature Engineering Controls
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### 4. Preprocessing Pipeline &amp; Clustering Configuration")

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        impute_strategy = st.selectbox(
            "Missing Value Imputation",
            options=["median", "mean", "zero"],
            index=0,
            help="Statistical strategy for imputing unrecorded numerical fields.",
        )
    with p_col2:
        scaler_choice = st.selectbox(
            "Feature Standardisation",
            options=["standard", "robust", "minmax"],
            index=0,
            help="StandardScaler: Zero mean unit variance. RobustScaler: Resilient to severe casualty outliers.",
        )
    with p_col3:
        cluster_k = st.slider(
            "Number of Clusters (K)",
            min_value=2,
            max_value=10,
            value=st.session_state.get("cluster_k", 4),
            step=1,
            help="Granularity of unsupervised hotspot partitioning.",
        )

    # Feature selector
    st.markdown("##### Select Safety Features for Unsupervised Clustering:")
    mapped_df_temp = apply_column_mapping(df_current_raw, user_mapping)
    
    # Potential clustering features
    candidate_features = [
        "Hour", "Number_of_Vehicles", "Number_of_Casualties",
        "Number_of_Fatalities", "Number_of_Injuries", "Speed_Limit",
        "Severity_Numeric", "Is_Night", "Is_Weekend", "Is_Adverse_Weather"
    ]
    
    selected_features = st.multiselect(
        "Clustering Features",
        options=candidate_features,
        default=[f for f in DEFAULT_CLUSTERING_FEATURES if f in candidate_features],
        help="Features used to compute distance matrices and cluster centroids.",
    )

    # 5. Prominent Run ML Button
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    if st.button("🚀 Apply Transformations & Run Unsupervised Machine Learning", use_container_width=True):
        with st.spinner("Executing data preprocessing, scaling, K-Means clustering, and PCA dimensionality reduction..."):
            mapped_df = apply_column_mapping(df_current_raw, user_mapping)
            _run_pipeline(
                df=mapped_df,
                k=cluster_k,
                impute_strategy=impute_strategy,
                scaler_type=scaler_choice,
                features=selected_features,
            )
            st.success(f"Analysis complete! Segmented {len(st.session_state['df_processed']):,} records into {cluster_k} clusters.")
            st.rerun()

    # 6. Audit summary of active model
    prep_summary = st.session_state.get("preprocessing_summary", None)
    if prep_summary:
        with st.expander("🔍 Preprocessing Audit Trail & Transformations Log", expanded=False):
            st.write(f"**Rows Processed:** {prep_summary['final_rows']:,} (Removed {prep_summary['rows_removed']} records)")
            st.write(f"**Features Utilized:** {', '.join(prep_summary['features_used'])}")
            st.write(f"**Standardisation Technique:** {prep_summary['scaler']}")
            st.markdown("##### Step-by-Step Transformations:")
            for log_item in prep_summary["audit_log"]:
                st.markdown(f"- {log_item}")


def _run_pipeline(
    df: pd.DataFrame,
    k: int = 4,
    impute_strategy: str = "median",
    scaler_type: str = "standard",
    features: list = None,
):
    """Executes the complete unsupervised machine learning pipeline and stores results in session_state."""
    features = features or DEFAULT_CLUSTERING_FEATURES
    preprocessor = AccidentPreprocessor(
        impute_strategy=impute_strategy,
        remove_duplicates=True,
        scaler_type=scaler_type,
        selected_features=features,
    )
    df_clean, X_scaled, prep_summary = preprocessor.fit_transform(df)

    # 1. K-Means
    km_results = ClusteringEngine.run_kmeans(X_scaled, n_clusters=k, random_state=42)
    df_clean["Cluster"] = km_results["labels"]

    # 2. PCA
    pca_results = PCAEngine.fit_transform(X_scaled, feature_names=prep_summary["features_used"])

    # 3. Profiling & Labels
    summary_df, cluster_meta = ClusterProfiler.generate_cluster_profiles(df_clean, cluster_col="Cluster")
    deviation_df = ClusterProfiler.compute_feature_deviations(
        df_clean, feature_names=prep_summary["features_used"], cluster_col="Cluster"
    )

    # Store in session state
    st.session_state["df_processed"] = df_clean
    st.session_state["X_scaled"] = X_scaled
    st.session_state["cluster_k"] = k
    st.session_state["clustering_results"] = km_results
    st.session_state["pca_results"] = pca_results
    st.session_state["cluster_summary_df"] = summary_df
    st.session_state["cluster_metadata"] = cluster_meta
    st.session_state["feature_deviations_df"] = deviation_df
    st.session_state["preprocessing_summary"] = prep_summary
