"""
KPI and Metric display components for Streamlit.
"""

from typing import Optional
import pandas as pd
import streamlit as st

from config.settings import SEGMENT_COLORS
from src.cleaning import CleaningReport
from src.personas import PERSONA_DEFINITIONS


def render_overview_kpis(
    labeled_df: pd.DataFrame,
    cleaning_report: Optional[CleaningReport] = None,
    optimal_k: Optional[int] = None,
):
    """
    Renders top KPI metric cards (Total Customers, Total Revenue, Average Customer Spend, Active Segments).
    """
    total_customers = len(labeled_df)
    total_revenue = labeled_df["monetary"].sum()
    avg_spend = total_revenue / total_customers if total_customers > 0 else 0.0
    num_segments = labeled_df["segment"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Customers",
            value=f"{total_customers:,}",
            help="Unique identified customers in the cleaned dataset",
        )

    with col2:
        st.metric(
            label="Total Revenue",
            value=f"${total_revenue:,.2f}",
            help="Sum of (quantity * unit_price) across valid completed transactions",
        )

    with col3:
        st.metric(
            label="Avg Customer Lifetime Spend",
            value=f"${avg_spend:,.2f}",
            help="Average total revenue generated per customer",
        )

    with col4:
        k_label = f"{num_segments} Segments"
        if optimal_k:
            k_label += f" (k={optimal_k})"
        st.metric(
            label="Segments Identified",
            value=k_label,
            help="Number of distinct behavioral clusters discovered by K-Means",
        )


def render_persona_cards_grid(labeled_df: pd.DataFrame):
    """
    Renders styled cards for each customer persona with strategic recommendations.
    """
    total_cust = len(labeled_df)
    total_rev = labeled_df["monetary"].sum()

    grouped = (
        labeled_df.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            revenue=("monetary", "sum"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .reset_index()
        .sort_values(by="revenue", ascending=False)
    )

    st.subheader("💡 Segment Playbooks & Strategic Action Plans")

    # Render in 2 columns of cards
    cols = st.columns(2)

    for i, row in grouped.iterrows():
        seg_name = row["segment"]
        c_count = row["customers"]
        c_pct = (c_count / total_cust * 100) if total_cust > 0 else 0
        r_sum = row["revenue"]
        r_pct = (r_sum / total_rev * 100) if total_rev > 0 else 0

        persona_info = PERSONA_DEFINITIONS.get(
            seg_name,
            {
                "description": "Distinct behavioral customer segment.",
                "action": "Engage based on recency and spending pattern.",
                "color": SEGMENT_COLORS.get(seg_name, "#4A90E2"),
            },
        )

        col = cols[i % 2]
        with col:
            with st.container(border=True):
                color_dot = f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background-color:{persona_info['color']};margin-right:8px;'></span>"
                st.markdown(f"### {color_dot}{seg_name}", unsafe_allow_html=True)
                
                # Metrics bar
                m1, m2, m3 = st.columns(3)
                m1.caption(f"**Customers:** {c_count:,} ({c_pct:.1f}%)")
                m2.caption(f"**Revenue:** ${r_sum:,.0f} ({r_pct:.1f}%)")
                m3.caption(f"**Avg Spend:** ${row['avg_monetary']:,.1f}")

                st.markdown(f"**Profile:** {persona_info['description']}")
                st.info(f"**🎯 Recommended Action:** {persona_info['action']}")
