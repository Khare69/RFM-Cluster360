"""
Sidebar filtering component across all dashboard views.
"""

from typing import Tuple
import pandas as pd
import streamlit as st


def render_sidebar_filters(
    labeled_df: pd.DataFrame,
    key_prefix: str = "filter",
) -> pd.DataFrame:
    """
    Renders filter controls in the sidebar and returns the filtered DataFrame.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Dataset with 'segment', 'monetary', and optional 'country'.
    key_prefix : str
        Prefix for Streamlit widget keys to prevent key collisions across pages.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame subset.
    """
    if labeled_df is None or labeled_df.empty:
        return labeled_df

    st.sidebar.markdown("### 🎯 Data Filters")

    # Segment filter
    all_segments = sorted(labeled_df["segment"].dropna().unique())
    selected_segments = st.sidebar.multiselect(
        "Filter by Segment",
        options=all_segments,
        default=all_segments,
        key=f"{key_prefix}_segments",
        help="Select one or more customer segments to display",
    )

    # Country filter if present
    has_country = "country" in labeled_df.columns and labeled_df["country"].nunique() > 1
    selected_countries = None
    if has_country:
        all_countries = sorted(labeled_df["country"].dropna().unique())
        selected_countries = st.sidebar.multiselect(
            "Filter by Country",
            options=all_countries,
            default=all_countries,
            key=f"{key_prefix}_countries",
        )

    # Monetary spend slider
    min_spend = float(labeled_df["monetary"].min())
    max_spend = float(labeled_df["monetary"].max())
    
    if min_spend < max_spend:
        diff = max_spend - min_spend
        step_val = min(max(0.01, round(diff / 100, 2)), diff)
        spend_range = st.sidebar.slider(
            "Filter by Total Spend ($)",
            min_value=float(min_spend),
            max_value=float(max_spend),
            value=(float(min_spend), float(max_spend)),
            step=float(step_val),
            key=f"{key_prefix}_spend_slider",
        )
    else:
        spend_range = (min_spend, max_spend)

    # Apply filters
    filtered_df = labeled_df.copy()
    if selected_segments:
        filtered_df = filtered_df[filtered_df["segment"].isin(selected_segments)]
    else:
        filtered_df = filtered_df.iloc[0:0]  # Empty if nothing selected

    if has_country and selected_countries:
        filtered_df = filtered_df[filtered_df["country"].isin(selected_countries)]

    filtered_df = filtered_df[
        (filtered_df["monetary"] >= spend_range[0]) & (filtered_df["monetary"] <= spend_range[1])
    ]

    # Show count in sidebar
    st.sidebar.caption(f"Showing **{len(filtered_df):,}** of **{len(labeled_df):,}** customers ({len(filtered_df)/len(labeled_df)*100:.1f}%)")

    return filtered_df
