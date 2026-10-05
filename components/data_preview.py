"""
Data preview and transparent cleaning report UI component.
"""

from typing import Dict, List, Optional
import pandas as pd
import streamlit as st

from config.settings import REQUIRED_COLUMNS
from src.cleaning import CleaningReport


def render_column_mapping_ui(
    df: pd.DataFrame,
    detected_mapping: Dict[str, Optional[str]],
) -> Dict[str, str]:
    """
    Renders an interactive UI where the user can verify and adjust column mappings.

    Parameters
    ----------
    df : pd.DataFrame
        Raw uploaded dataframe.
    detected_mapping : dict
        Canonical name -> detected column name.

    Returns
    -------
    Dict[str, str]
        User-confirmed mapping.
    """
    st.markdown("#### 🔗 Step 1: Verify Column Mapping")
    st.markdown(
        "We automatically scanned your column headers. "
        "Please confirm or adjust the mappings below to ensure proper analysis:"
    )

    available_cols = ["-- Select Column --"] + list(df.columns)
    confirmed_mapping = {}

    cols = st.columns(len(REQUIRED_COLUMNS))

    canonical_labels = {
        "customer_id": "Customer ID *",
        "invoice_no": "Invoice / Order # *",
        "invoice_date": "Transaction Date *",
        "quantity": "Quantity *",
        "unit_price": "Unit Price ($) *",
    }

    for i, canonical in enumerate(REQUIRED_COLUMNS):
        with cols[i]:
            detected = detected_mapping.get(canonical)
            default_idx = available_cols.index(detected) if detected in available_cols else 0
            
            chosen = st.selectbox(
                label=canonical_labels.get(canonical, canonical),
                options=available_cols,
                index=default_idx,
                key=f"mapping_{canonical}",
                help=f"Matches your CSV column to the required {canonical} attribute.",
            )
            confirmed_mapping[canonical] = chosen if chosen != "-- Select Column --" else None

    # Optional columns (Country, Description)
    with st.expander("Optional Column Mappings (Country, Description)"):
        opt_cols = st.columns(2)
        with opt_cols[0]:
            opt_country = detected_mapping.get("country")
            c_idx = available_cols.index(opt_country) if opt_country in available_cols else 0
            country_chosen = st.selectbox("Country (Optional)", options=available_cols, index=c_idx, key="mapping_country")
            if country_chosen != "-- Select Column --":
                confirmed_mapping["country"] = country_chosen
        with opt_cols[1]:
            opt_desc = detected_mapping.get("description")
            d_idx = available_cols.index(opt_desc) if opt_desc in available_cols else 0
            desc_chosen = st.selectbox("Description (Optional)", options=available_cols, index=d_idx, key="mapping_description")
            if desc_chosen != "-- Select Column --":
                confirmed_mapping["description"] = desc_chosen

    return confirmed_mapping


def render_cleaning_report_card(report: CleaningReport):
    """
    Renders transparent cleaning summary and before/after metrics.
    """
    st.markdown("#### 🧹 Step 2: Data Cleaning & Hygiene Audit")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Raw Rows Uploaded", f"{report.total_rows_before:,}")
    with col2:
        st.metric("Valid Rows Retained", f"{report.total_rows_after:,}", f"{report.retention_percentage}% retained")
    with col3:
        st.metric("Cleaned Customers", f"{report.total_customers:,}")
    with col4:
        st.metric("Cleaned Revenue", f"${report.total_revenue:,.2f}")

    # Itemized removal breakdown
    if report.total_rows_removed > 0:
        with st.expander("🔎 View Itemized Cleaning Breakdown (Why rows were filtered out)"):
            breakdown_data = [
                {"Reason for Removal": "Missing / Guest Customer ID", "Rows Filtered": report.rows_removed_missing_customer, "Why": "Guest checkouts without an account cannot be tracked across orders"},
                {"Reason for Removal": "Cancelled Orders (Invoice starts with 'C')", "Rows Filtered": report.rows_removed_cancelled, "Why": "Cancelled orders do not represent completed purchases"},
                {"Reason for Removal": "Missing / Blank Invoice Number", "Rows Filtered": report.rows_removed_missing_invoice, "Why": "Transactions without an invoice identifier cannot be attributed to an order"},
                {"Reason for Removal": "Non-Positive Quantity (≤ 0)", "Rows Filtered": report.rows_removed_non_positive_qty, "Why": "Returns, adjustments, or corrupted counts"},
                {"Reason for Removal": "Non-Positive Price (≤ 0)", "Rows Filtered": report.rows_removed_non_positive_price, "Why": "Free samples, giveaways, or zero-price test transactions"},
                {"Reason for Removal": "Invalid / Unparseable Dates", "Rows Filtered": report.rows_removed_invalid_date, "Why": "Cannot determine recency without valid timestamps"},
            ]
            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

            if report.duplicate_rows_detected > 0:
                st.caption(f"ℹ️ Found **{report.duplicate_rows_detected}** exact duplicate rows in the dataset.")
