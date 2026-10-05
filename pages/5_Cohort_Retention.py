"""
Page 5: Cohort Retention Analysis — Monthly Acquisition Cohorts & Retention Heatmap.
"""

import streamlit as st

from src.cohort import get_cohort_summary_stats
from viz.cohort_heatmap import cohort_retention_heatmap

st.set_page_config(page_title="Cohort Retention | RFM Cluster360", page_icon="📈", layout="wide")

# Guard clause
if "cohort_matrix" not in st.session_state or st.session_state.cohort_matrix is None:
    st.warning("⚠️ No dataset loaded. Please go to the **Home** page to upload a CSV or load the sample data.")
    if st.button("⬅️ Return to Home"):
        st.switch_page("app.py")
    st.stop()

st.title("📈 Monthly Customer Cohort Retention")
st.markdown(
    "Track customer stickiness over time. Customers are grouped by the month they placed their **first order** (their cohort). "
    "Each cell indicates what percentage of those customers returned to purchase in subsequent months."
)

retention_matrix = st.session_state.cohort_matrix
counts_matrix = st.session_state.cohort_counts

# Summary KPIs
cohort_kpis = get_cohort_summary_stats(retention_matrix)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Cohorts Tracked", f"{cohort_kpis['total_cohorts']}")
with col2:
    st.metric("Avg Month 1 Retention", f"{cohort_kpis['avg_m1_retention']}%", help="Percentage of new customers who return to make an order in their 2nd month")
with col3:
    st.metric("Avg Month 3 Retention", f"{cohort_kpis['avg_m3_retention']}%", help="Retention 3 months post-acquisition")
with col4:
    st.metric("Avg Month 6 Retention", f"{cohort_kpis['avg_m6_retention']}%", help="Retention 6 months post-acquisition")

st.markdown("---")

# Retention Heatmap
st.subheader("🔥 Retention Heatmap Matrix")
fig_cohort = cohort_retention_heatmap(retention_matrix, counts_matrix)
st.plotly_chart(fig_cohort, use_container_width=True)

st.markdown("---")

# Strategy and interpretation guide
with st.expander("📚 How to Interpret This Heatmap for Business Growth"):
    st.markdown("""
    - **Read Across a Row (Customer Lifecycle):** Follow a single cohort from left to right to see how quickly customer activity drops off after acquisition. A steep drop from Month 0 to Month 1 signals an onboarding or product expectation issue.
    - **Read Down a Column (Cohort Quality Comparison):** Compare the same month index across different cohort rows. If newer rows have higher retention percentages than older rows in Month 1, your recent marketing and product improvements are working!
    - **Identify the Churn Plateau:** Look for the month where retention curves flatten out. That baseline represents your core loyal customer base.
    """)
