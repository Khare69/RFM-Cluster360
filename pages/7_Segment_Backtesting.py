"""
Page 7: Out-of-Time Segment Backtesting & Migration Analysis.
Evaluates model stability and forward predictive validity across temporal observation windows.
"""

import pandas as pd
import streamlit as st

from config.settings import DEFAULT_CURRENCY_SYMBOL
from src.backtesting import run_backtesting_pipeline
from viz.backtesting_charts import (
    render_forward_spend_comparison,
    render_segment_retention_rates,
    render_transition_heatmap,
)

st.set_page_config(page_title="Segment Backtesting | RFM Cluster360", page_icon="⏳", layout="wide")

# Guard clause
if "cleaned_df" not in st.session_state or st.session_state.cleaned_df is None:
    st.warning("⚠️ No dataset loaded. Please go to the **Home** page to upload a CSV or load the sample data.")
    if st.button("⬅️ Return to Home"):
        st.switch_page("app.py")
    st.stop()

raw_df = st.session_state.cleaned_df
currency_symbol = st.session_state.get("currency_symbol", DEFAULT_CURRENCY_SYMBOL)

st.title("⏳ Out-of-Time Segment Backtesting & Migration")
st.markdown(
    "Validate the predictive stability of customer personas across time. "
    "Split transaction records into a **Calibration Window ($T_1$)** and a forward **Outcome Window ($T_2$)** "
    "to observe how customer segments migrate, identify churn plateaus, and measure actual forward revenue realization."
)

# Date calculations
invoice_dates = pd.to_datetime(raw_df["invoice_date"])
min_date = invoice_dates.min().date()
max_date = invoice_dates.max().date()
total_days = (max_date - min_date).days

if total_days < 7:
    st.warning(
        f"⚠️ Dataset date span ({total_days} days) is too short for temporal backtesting. "
        "A minimum observation span of at least 7 days is required."
    )
    st.stop()

# Interactive controls
st.sidebar.header("⏳ Backtesting Parameters")

# Default cutoff to midpoint
midpoint_date = (min_date + pd.Timedelta(days=total_days // 2))

cutoff_selection = st.sidebar.date_input(
    "Temporal Cutoff Date (T1 / T2 Split)",
    value=midpoint_date,
    min_value=min_date + pd.Timedelta(days=2),
    max_value=max_date - pd.Timedelta(days=2),
    help="Transactions on or before this date form Calibration Window T1. Later transactions form Outcome Window T2.",
)

curr_k = st.session_state.get("k", 4)
selected_k = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=8, value=curr_k, step=1)

# Run Backtesting Analysis
try:
    with st.spinner("Executing out-of-time temporal window split and Markov transition modeling..."):
        results = run_backtesting_pipeline(
            df=raw_df,
            cutoff_date=cutoff_selection,
            k=selected_k,
            random_state=42,
        )
except Exception as e:
    st.error(f"Error computing backtesting analysis: {e}")
    st.stop()

# 1. Summary KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "T1 Calibration Cohort",
        f"{results['total_t1_customers']:,} cust",
        help="Unique customers who made at least one purchase in Calibration Window T1.",
    )
with col2:
    st.metric(
        "T2 Repeat Buyers",
        f"{results['retained_count']:,} cust",
        help="Customers active in T1 who placed another order during Outcome Window T2.",
    )
with col3:
    st.metric(
        "Forward Retention Rate",
        f"{results['retention_rate'] * 100:.1f}%",
        help="Percentage of T1 customers who transacted again in T2.",
    )
with col4:
    tot_fwd_spend = results["forward_metrics"]["total_forward_spend"].sum()
    st.metric(
        "Realized T2 Revenue",
        f"{currency_symbol}{tot_fwd_spend:,.0f}",
        help="Total forward revenue generated in T2 strictly by the T1 calibration cohort.",
    )

st.markdown("---")

# 2. Markov Transition Heatmap
st.subheader("🔥 Segment Migration Matrix (Markov Probability Heatmap)")
st.caption(
    "Rows represent the baseline persona established in $T_1$. "
    "Columns represent the destination persona in $T_2$ (including **Churned** for customers with zero $T_2$ orders). "
    "Each cell indicates the empirical transition probability $P(T_2 \\mid T_1)$ and customer count."
)

fig_heatmap = render_transition_heatmap(results["prob_matrix"], results["counts_matrix"])
st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("---")

# 3. Forward Realization Charts
st.subheader("📊 Empirical Predictive Validation")
st.caption(
    "Proves whether high-tier segments (e.g., Champions, Loyal Customers) consistently deliver "
    "superior forward spend and retention in subsequent periods."
)

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    fig_spend = render_forward_spend_comparison(results["forward_metrics"], currency_symbol=currency_symbol)
    st.plotly_chart(fig_spend, use_container_width=True)

with chart_col2:
    fig_ret = render_segment_retention_rates(results["forward_metrics"])
    st.plotly_chart(fig_ret, use_container_width=True)

st.markdown("---")

# 4. Customer-Level Migration Drilldown Table
st.subheader("👥 Customer Migration Drilldown")

migration_df = results["migration_details"]

f_col1, f_col2 = st.columns(2)
with f_col1:
    available_t1_segs = ["All"] + sorted(migration_df["t1_segment"].unique().tolist())
    selected_t1_filter = st.selectbox("Filter by Initial T1 Segment:", options=available_t1_segs)

with f_col2:
    available_statuses = ["All"] + sorted(migration_df["migration_status"].unique().tolist())
    selected_status_filter = st.selectbox("Filter by Migration Status:", options=available_statuses)

filtered_migration = migration_df.copy()
if selected_t1_filter != "All":
    filtered_migration = filtered_migration[filtered_migration["t1_segment"] == selected_t1_filter]
if selected_status_filter != "All":
    filtered_migration = filtered_migration[filtered_migration["migration_status"] == selected_status_filter]

st.dataframe(
    filtered_migration.rename(columns={
        "customer_id": "Customer ID",
        "t1_segment": "T1 Baseline Segment",
        "t2_segment": "T2 Outcome Segment",
        "t1_spend": f"T1 Spend ({currency_symbol})",
        "t2_spend": f"T2 Spend ({currency_symbol})",
        "migration_status": "Status",
    }),
    use_container_width=True,
    hide_index=True,
)

csv_data = filtered_migration.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Migration Drilldown (CSV)",
    data=csv_data,
    file_name=f"rfm_segment_migration_{cutoff_selection}.csv",
    mime="text/csv",
)

st.markdown("---")

# 5. Strategic Guide
with st.expander("📚 How to Interpret Out-of-Time Backtesting for Business Strategy"):
    st.markdown("""
    - **Diagonal Dominance (Model Stability):** In the transition heatmap, a strong dark diagonal from top-left indicates high segment stability (customers staying in their persona).
    - **Champion Churn Leakage:** If a substantial percentage of your *Champions* transition directly into *Churned (Inactive)* rather than decaying into *At Risk*, investigate post-purchase customer satisfaction or delivery bottlenecks.
    - **Forward Revenue Monotonicity:** In a well-specified segmentation, the total forward revenue in $T_2$ should decline monotonically as you move from *Champions* down to *Lost* or *At Risk*.
    - **Reactivation Opportunity:** Analyze customers in low-tier $T_1$ segments who upgraded to *Champions* or *Loyal* in $T_2$ to identify successful marketing campaign triggers.
    """)
