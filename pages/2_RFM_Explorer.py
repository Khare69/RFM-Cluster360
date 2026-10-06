"""
Page 2: RFM Explorer — Metric Distributions and Machine Learning Model Diagnostics.
"""

import pandas as pd
import streamlit as st

from components.filters import render_sidebar_filters
from src.clustering import get_cluster_profiles
from src.rfm import get_rfm_summary_stats
from viz.rfm_charts import (
    elbow_curve,
    rfm_histograms,
    segment_radar_comparison,
    silhouette_plot,
)

st.set_page_config(page_title="RFM Explorer | RFM Cluster360", page_icon="🔍", layout="wide")

# Guard clause
if "labeled_df" not in st.session_state or st.session_state.labeled_df is None:
    st.warning("⚠️ No dataset loaded. Please go to the **Home** page to upload a CSV or load the sample data.")
    if st.button("⬅️ Return to Home"):
        st.switch_page("app.py")
    st.stop()

st.title("🔍 RFM Explorer & ML Diagnostics")
st.markdown(
    "Deep dive into the underlying Recency, Frequency, and Monetary distributions and explore the machine learning model selection process."
)

filtered_df = render_sidebar_filters(st.session_state.labeled_df, key_prefix="rfm_exp")

if filtered_df.empty:
    st.warning("No customers match the current filter selection.")
    st.stop()

tab_dist, tab_ml, tab_dna = st.tabs([
    "📊 Metric Distributions",
    "🔬 Clustering Diagnostics (Elbow & Silhouette)",
    "🧬 Segment DNA & Profiles",
])

with tab_dist:
    st.subheader("Distribution of R, F, and M Metrics")
    st.markdown(
        "Observe the natural skewness across your customer base. Monetary spend and order frequency typically follow an 80/20 power-law distribution."
    )
    fig_hist = rfm_histograms(filtered_df)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### 📐 Statistical Summary Percentiles")
    stats = get_rfm_summary_stats(filtered_df)
    stats_df = pd.DataFrame(stats).T
    stats_df.index.name = "Metric"
    st.dataframe(stats_df, use_container_width=True)

with tab_ml:
    st.subheader("How the Algorithm Chose the Number of Clusters (k)")
    st.markdown(
        "To avoid arbitrary guessing, RFM Cluster360 automatically tests a range of cluster counts ($k=2$ through $8$) "
        "and evaluates both **Inertia (Compactness)** and the **Silhouette Score (Separation Quality)**."
    )

    eval_metrics = st.session_state.clustering_metrics
    if eval_metrics:
        if eval_metrics.get("warning"):
            st.warning(f"⚠️ {eval_metrics['warning']}")
        col1, col2 = st.columns(2)
        with col1:
            fig_elbow = elbow_curve(eval_metrics)
            st.plotly_chart(fig_elbow, use_container_width=True)
            st.caption(
                "📉 **Elbow Method:** Measures within-cluster sum of squares (WCSS). We look for an 'elbow' where adding more clusters yields diminishing returns in compactness."
            )

        with col2:
            fig_sil = silhouette_plot(eval_metrics)
            st.plotly_chart(fig_sil, use_container_width=True)
            st.caption(
                "✨ **Silhouette Score:** Quantifies how well-separated the clusters are (−1 to +1). Peak score represents the most distinct customer groupings."
            )

        best_k = eval_metrics["best_k"]
        best_score = eval_metrics["best_silhouette"]
        st.success(
            f"🤖 **Auto-Recommendation:** $k={best_k}$ achieved the highest silhouette score of **{best_score:.3f}**, indicating strong separation between customer groups."
        )

with tab_dna:
    st.subheader("Segment Attribute Comparison (DNA Radar)")
    st.markdown("Compares the relative strengths and behavioral signatures of each segment.")
    
    fig_radar = segment_radar_comparison(filtered_df)
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("#### 📋 Cluster Centroid Profiles")
    profiles = get_cluster_profiles(filtered_df)
    st.dataframe(profiles, use_container_width=True, hide_index=True)
