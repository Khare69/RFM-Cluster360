"""
Cohort retention analysis module.
Calculates acquisition cohorts and retention percentage matrices across time periods.
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


def build_cohort_matrix(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs cohort retention percentage matrix and raw customer count matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction DataFrame with 'customer_id' and 'invoice_date'.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - retention_matrix : DataFrame of retention percentages (0.0 to 100.0)
        - counts_matrix : DataFrame of raw active customer counts per cohort and period
    """
    if df.empty or "customer_id" not in df.columns or "invoice_date" not in df.columns:
        raise ValueError("DataFrame must contain 'customer_id' and 'invoice_date'.")

    work_df = df.copy()
    work_df["invoice_date"] = pd.to_datetime(work_df["invoice_date"])

    # Extract transaction month
    work_df["transaction_month"] = work_df["invoice_date"].dt.to_period("M")

    # Determine cohort month (month of first purchase)
    work_df["cohort_month"] = work_df.groupby("customer_id")["transaction_month"].transform("min")

    # Calculate period index (months elapsed since acquisition)
    # E.g. period 0 is acquisition month, period 1 is 1 month later
    def _month_diff(d1, d2):
        return (d1.year - d2.year) * 12 + (d1.month - d2.month)

    work_df["cohort_index"] = _month_diff(
        work_df["transaction_month"].dt,
        work_df["cohort_month"].dt,
    )

    # Count unique customers per cohort and period index
    cohort_data = (
        work_df.groupby(["cohort_month", "cohort_index"])["customer_id"]
        .nunique()
        .reset_index()
    )

    # Pivot into matrix
    counts_matrix = cohort_data.pivot(
        index="cohort_month",
        columns="cohort_index",
        values="customer_id",
    )

    # Reindex columns to contiguous range 0..max_period to avoid skipping quiet months
    if not counts_matrix.empty and len(counts_matrix.columns) > 0:
        max_period = int(counts_matrix.columns.max())
        counts_matrix = counts_matrix.reindex(columns=range(0, max_period + 1))

    # Format cohort month index as readable string (e.g., '2023-01')
    counts_matrix.index = counts_matrix.index.strftime("%Y-%m")

    # Compute retention percentages relative to period 0
    cohort_sizes = counts_matrix.iloc[:, 0]
    retention_matrix = counts_matrix.divide(cohort_sizes, axis=0) * 100
    retention_matrix = retention_matrix.round(1)

    return retention_matrix, counts_matrix


def get_cohort_summary_stats(retention_matrix: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts high-level summary KPIs from the retention matrix.

    Parameters
    ----------
    retention_matrix : pd.DataFrame
        Matrix where rows are cohorts and columns are period indices.

    Returns
    -------
    dict
        Summary stats including avg month-1, month-3, month-6 retention.
    """
    stats = {
        "total_cohorts": len(retention_matrix),
        "avg_m1_retention": 0.0,
        "avg_m3_retention": 0.0,
        "avg_m6_retention": 0.0,
    }

    if 1 in retention_matrix.columns:
        m1 = retention_matrix[1].dropna()
        if not m1.empty:
            stats["avg_m1_retention"] = round(float(m1.mean()), 1)

    if 3 in retention_matrix.columns:
        m3 = retention_matrix[3].dropna()
        if not m3.empty:
            stats["avg_m3_retention"] = round(float(m3.mean()), 1)

    if 6 in retention_matrix.columns:
        m6 = retention_matrix[6].dropna()
        if not m6.empty:
            stats["avg_m6_retention"] = round(float(m6.mean()), 1)

    return stats
