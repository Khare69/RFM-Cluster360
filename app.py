"""
RFM Cluster360 — Main Application Entrypoint.
Orchestrates file ingestion, interactive column mapping, data cleaning, RFM calculation, and clustering pipeline.
"""

from datetime import datetime
import os
import pandas as pd
import streamlit as st

from config.settings import (
    DEFAULT_K_MAX,
    DEFAULT_K_MIN,
    MAX_FILE_SIZE_MB,
    MIN_CUSTOMERS_FOR_CLUSTERING,
)
from components.data_preview import render_cleaning_report_card, render_column_mapping_ui
from components.download_button import render_export_section
from src.cleaning import clean_transactions
from src.clustering import build_clustered_df, evaluate_clusters, preprocess_rfm, run_kmeans
from src.cohort import build_cohort_matrix
from src.ingestion import auto_map_columns, load_csv, validate_mapping
from src.personas import label_segments
from src.rfm import calculate_rfm, default_snapshot_date

# Configure Streamlit page layout and appearance
st.set_page_config(
    page_title="RFM Cluster360",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initializes default session state keys if not already present."""
    defaults = {
        "raw_df": None,
        "clean_df": None,
        "cleaning_report": None,
        "rfm_df": None,
        "clustered_df": None,
        "labeled_df": None,
        "clustering_metrics": None,
        "selected_k": None,
        "cluster_map": None,
        "cohort_matrix": None,
        "cohort_counts": None,
        "snapshot_date": None,
        "is_sample_data": False,
        "file_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_analysis():
    """Clears all cached analysis data to start fresh."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()
    st.rerun()


def run_pipeline(
    raw_df: pd.DataFrame,
    mapping: dict,
    k_override: int = None,
    custom_snapshot_date=None,
    remove_duplicates: bool = False,
):
    """Executes the full end-to-end segmentation pipeline."""
    with st.spinner("1/5 Cleaning transaction records and filtering anomalies..."):
        clean_df, report = clean_transactions(raw_df, mapping, remove_duplicates=remove_duplicates)

    if clean_df.empty or report.total_customers < MIN_CUSTOMERS_FOR_CLUSTERING:
        st.error(
            f"After cleaning, only {report.total_customers} valid customers remain. "
            f"At least {MIN_CUSTOMERS_FOR_CLUSTERING} customers are required for clustering."
        )
        return

    with st.spinner("2/5 Computing Recency, Frequency, and Monetary metrics..."):
        snap_dt = custom_snapshot_date if custom_snapshot_date else default_snapshot_date(clean_df)
        rfm_df = calculate_rfm(clean_df, snapshot_date=snap_dt)

    with st.spinner("3/5 Normalizing distributions and evaluating optimal cluster count..."):
        scaled_data, _, _ = preprocess_rfm(rfm_df)
        eval_metrics = evaluate_clusters(scaled_data)
        best_k = eval_metrics["best_k"]
        chosen_k = k_override if k_override is not None else best_k

    with st.spinner(f"4/5 Fitting K-Means with k={chosen_k} and applying persona labels..."):
        labels, _, _ = run_kmeans(scaled_data, k=chosen_k)
        clustered_df = build_clustered_df(rfm_df, labels)
        labeled_df, cluster_map = label_segments(clustered_df)

    with st.spinner("5/5 Generating monthly cohort retention matrix..."):
        retention_matrix, counts_matrix = build_cohort_matrix(clean_df)

    # Persist in session state
    st.session_state.raw_df = raw_df
    st.session_state.clean_df = clean_df
    st.session_state.cleaning_report = report
    st.session_state.rfm_df = rfm_df
    st.session_state.clustered_df = clustered_df
    st.session_state.labeled_df = labeled_df
    st.session_state.clustering_metrics = eval_metrics
    st.session_state.selected_k = chosen_k
    st.session_state.cluster_map = cluster_map
    st.session_state.cohort_matrix = retention_matrix
    st.session_state.cohort_counts = counts_matrix
    st.session_state.snapshot_date = snap_dt


def main():
    init_session_state()

    # Sidebar Header & Controls
    st.sidebar.image(
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400&auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
        use_container_width=True,
    )
    st.sidebar.title("🎯 RFM Cluster360")
    st.sidebar.caption("Machine-Learning Customer Segmentation Engine")

    # If data already loaded, provide quick status & reset option
    if st.session_state.labeled_df is not None:
        st.sidebar.success(
            f"✅ **Analysis Active**\n\n"
            f"• **Customers:** {len(st.session_state.labeled_df):,}\n\n"
            f"• **Segments:** {st.session_state.selected_k} (k={st.session_state.selected_k})\n\n"
            f"• **Revenue:** ${st.session_state.labeled_df['monetary'].sum():,.2f}"
        )
        if st.sidebar.button("🔄 Reset / Upload New Dataset", use_container_width=True):
            reset_analysis()

        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Model Parameters")
        
        # Allow user to tweak k on the fly
        curr_k = st.session_state.selected_k
        eval_metrics = st.session_state.clustering_metrics
        best_k = eval_metrics["best_k"] if eval_metrics else curr_k
        sil_text = f" ({eval_metrics['best_silhouette']:.3f})" if (eval_metrics and "best_silhouette" in eval_metrics) else ""

        min_k_val = DEFAULT_K_MIN
        max_k_val = min(DEFAULT_K_MAX, max(DEFAULT_K_MIN, len(st.session_state.rfm_df) - 1))

        if min_k_val < max_k_val:
            new_k = st.sidebar.slider(
                "Cluster Count (k)",
                min_value=min_k_val,
                max_value=max_k_val,
                value=min(max(curr_k, min_k_val), max_k_val),
                help=f"Auto-selected k={best_k} based on highest silhouette score{sil_text}.",
            )
        else:
            new_k = min_k_val
            st.sidebar.info(f"k is fixed at {min_k_val} for this dataset size")
        if new_k != curr_k:
            if st.sidebar.button("Apply New k", type="primary", use_container_width=True):
                scaled_data, _, _ = preprocess_rfm(st.session_state.rfm_df)
                labels, _, _ = run_kmeans(scaled_data, k=new_k)
                clustered_df = build_clustered_df(st.session_state.rfm_df, labels)
                labeled_df, cluster_map = label_segments(clustered_df)
                st.session_state.clustered_df = clustered_df
                st.session_state.labeled_df = labeled_df
                st.session_state.selected_k = new_k
                st.session_state.cluster_map = cluster_map
                st.rerun()

    # Main Page Title & Intro
    st.title("🎯 RFM Cluster360 — Customer Intelligence")
    st.markdown(
        "Transform raw retail transaction exports into actionable customer personas using **Recency, Frequency, and Monetary (RFM)** modeling and **Unsupervised Machine Learning (K-Means)**."
    )

    # If no dataset loaded, show Ingestion Interface
    if st.session_state.labeled_df is None:
        tab_upload, tab_sample = st.tabs(["📁 Upload Your CSV", "🧪 Try Sample Dataset"])

        raw_df = None
        file_name = None

        with tab_upload:
            st.markdown(f"Upload any standard transaction export (CSV format, max {MAX_FILE_SIZE_MB}MB).")
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=["csv"],
                help="Requires Customer ID, Invoice/Order #, Date, Quantity, and Unit Price.",
            )
            if uploaded_file is not None:
                with st.spinner("Reading file..."):
                    df_parsed, err = load_csv(uploaded_file, max_size_mb=MAX_FILE_SIZE_MB)
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        raw_df = df_parsed
                        file_name = uploaded_file.name
                        st.session_state.is_sample_data = False
                        st.session_state.temp_raw_df = raw_df
                        st.session_state.temp_file_name = file_name

        with tab_sample:
            st.markdown(
                "Don't have a transaction export ready? "
                "Explore with our curated **Retail E-Commerce Sample Dataset** (~1,200 transactions across 120 customers with realistic order patterns, returns, and guest checkouts)."
            )
            if st.button("🚀 Load Sample Retail Dataset", type="primary"):
                sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_retail.csv")
                if os.path.exists(sample_path):
                    df_parsed, err = load_csv(sample_path)
                    if not err:
                        raw_df = df_parsed
                        file_name = "sample_retail.csv"
                        st.session_state.is_sample_data = True
                        st.session_state.temp_raw_df = raw_df
                        st.session_state.temp_file_name = file_name
                else:
                    st.error("Sample dataset file not found.")

        if raw_df is None and "temp_raw_df" in st.session_state:
            raw_df = st.session_state.temp_raw_df
            file_name = st.session_state.temp_file_name

        if raw_df is not None:
            st.success(f"Loaded **{len(raw_df):,} rows** and **{len(raw_df.columns)} columns** from `{file_name}`.")

            # Column auto-mapping
            auto_detected = auto_map_columns(raw_df)
            user_mapping = render_column_mapping_ui(raw_df, auto_detected)

            is_valid, missing_cols = validate_mapping(user_mapping)

            st.markdown("---")
            col_left, col_right = st.columns([1, 1])

            with col_left:
                remove_dup = st.checkbox("Remove exact duplicate transactions", value=False)
            
            with col_right:
                if is_valid:
                    if st.button("✨ Run Customer Segmentation Pipeline", type="primary", use_container_width=True):
                        run_pipeline(
                            raw_df=raw_df,
                            mapping=user_mapping,
                            remove_duplicates=remove_dup,
                        )
                        st.rerun()
                else:
                    st.warning(f"⚠️ Please map the remaining required columns: **{', '.join(missing_cols)}**")

    # If dataset is loaded and pipeline has executed:
    else:
        st.success("🎉 **Customer Segmentation Pipeline Complete!**")
        
        # Display Data Hygiene Audit
        if st.session_state.cleaning_report:
            render_cleaning_report_card(st.session_state.cleaning_report)

        st.markdown("---")
        st.subheader("🧭 Next: Explore Your Dashboards")
        st.markdown(
            "Navigate using the **sidebar pages** on the left to explore deep analytics:"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("📊 **1. Overview**\n\nHigh-level segment metrics, revenue distribution, and strategic marketing playbooks.")
        with col2:
            st.info("🔍 **2. RFM Explorer**\n\nMetric distributions, elbow method, and silhouette score diagnostics.")
        with col3:
            st.info("🗺️ **3. 3D Cluster Map**\n\nInteractive 3D customer space with rotatable camera controls.")

        col4, col5, _ = st.columns(3)
        with col4:
            st.info("👤 **4. Customer Search**\n\nLook up individual customer profiles, purchase history timelines, and percentiles.")
        with col5:
            st.info("📈 **5. Cohort Retention**\n\nMonthly acquisition cohort retention heatmaps.")

        st.markdown("---")
        st.subheader("📋 Segmented Customers Preview")
        st.dataframe(
            st.session_state.labeled_df[
                ["customer_id", "segment", "recency", "frequency", "monetary", "first_purchase", "last_purchase"]
            ].head(100),
            use_container_width=True,
            hide_index=True,
        )

        # Export Component
        render_export_section(st.session_state.labeled_df)


if __name__ == "__main__":
    main()
