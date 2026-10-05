"""
Page 1: Overview Dashboard — Executive Summary, Segment Breakdown, and Strategic Playbooks.
"""

import streamlit as st

from components.download_button import render_export_section
from components.filters import render_sidebar_filters
from components.metric_cards import render_overview_kpis, render_persona_cards_grid
from viz.overview_charts import (
    rfm_summary_scatter,
    segment_customer_bar,
    segment_revenue_bar,
    segment_treemap,
)

st.set_page_config(page_title="Overview | RFM Cluster360", page_icon="📊", layout="wide")

# Guard clause: ensure data is loaded
if "labeled_df" not in st.session_state or st.session_state.labeled_df is None:
    st.warning("⚠️ No dataset loaded. Please go to the **Home** page to upload a CSV or load the sample data.")
    if st.button("⬅️ Return to Home"):
        st.switch_page("app.py")
    st.stop()

st.title("📊 Executive Overview & Customer Segments")
st.markdown("High-level customer distribution, revenue contribution, and strategic marketing playbooks.")

# Apply sidebar filters
filtered_df = render_sidebar_filters(st.session_state.labeled_df, key_prefix="ov")

if filtered_df.empty:
    st.warning("No customers match the current filter selection. Please adjust the sidebar filters.")
    st.stop()

# Top KPIs
render_overview_kpis(
    labeled_df=filtered_df,
    cleaning_report=st.session_state.cleaning_report,
    optimal_k=st.session_state.selected_k,
)

st.markdown("---")

# Chart Row 1: Customer Volume and Revenue by Segment
col1, col2 = st.columns(2)
with col1:
    fig_cust = segment_customer_bar(filtered_df)
    st.plotly_chart(fig_cust, use_container_width=True)

with col2:
    fig_rev = segment_revenue_bar(filtered_df)
    st.plotly_chart(fig_rev, use_container_width=True)

# Chart Row 2: Treemap and Value Landscape
col3, col4 = st.columns(2)
with col3:
    fig_tree = segment_treemap(filtered_df)
    st.plotly_chart(fig_tree, use_container_width=True)

with col4:
    fig_scatter = rfm_summary_scatter(filtered_df)
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# Strategic Action Plans
render_persona_cards_grid(filtered_df)

st.markdown("---")

# Export Component
render_export_section(filtered_df, title="📥 Export Filtered Customer Segment Data")
