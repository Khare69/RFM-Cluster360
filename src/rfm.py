"""
RFM (Recency, Frequency, Monetary) metric calculation module.
Pure pandas operations, completely unit-testable without database or UI dependencies.
"""

from datetime import date, datetime
from typing import Optional, Union, Dict, Any
import numpy as np
import pandas as pd


def default_snapshot_date(df: pd.DataFrame) -> pd.Timestamp:
    """
    Computes the default snapshot date as one day after the latest transaction date in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with an 'invoice_date' column.

    Returns
    -------
    pd.Timestamp
        Snapshot timestamp at midnight.
    """
    if df.empty or "invoice_date" not in df.columns:
        raise ValueError("DataFrame is empty or missing 'invoice_date' column.")
    
    max_date = pd.to_datetime(df["invoice_date"]).max()
    if pd.isna(max_date):
        raise ValueError("No valid dates found in 'invoice_date' column.")
    
    # Snap to midnight of the following day
    return pd.Timestamp(max_date.date()) + pd.Timedelta(days=1)


def calculate_rfm(
    df: pd.DataFrame,
    snapshot_date: Optional[Union[str, date, datetime, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Calculates RFM metrics per unique customer.

    Definitions:
    - Recency: Days elapsed between the snapshot date and customer's latest purchase.
    - Frequency: Total number of unique orders/invoices placed by the customer.
    - Monetary: Total monetary expenditure across all purchases (quantity * unit_price).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction DataFrame containing ['customer_id', 'invoice_no', 'invoice_date', 'quantity', 'unit_price'].
    snapshot_date : str, date, datetime, optional
        Reference point date for calculating recency. Defaults to 1 day after max(invoice_date).

    Returns
    -------
    pd.DataFrame
        One row per customer with columns:
        ['customer_id', 'recency', 'frequency', 'monetary', 'first_purchase', 'last_purchase', ('country' if present)]
    """
    required = ["customer_id", "invoice_no", "invoice_date", "quantity", "unit_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for RFM calculation: {missing}")

    if df.empty:
        raise ValueError("Cannot calculate RFM on an empty DataFrame.")

    work_df = df.copy()
    # Filter out null or blank invoice_no defensively
    if "invoice_no" in work_df.columns:
        inv_valid = work_df["invoice_no"].notna() & (~work_df["invoice_no"].astype(str).str.strip().isin(["", "nan", "none", "null"]))
        work_df = work_df[inv_valid]

    # Normalize to tz-naive to prevent subtraction mismatches
    parsed_dates = pd.to_datetime(work_df["invoice_date"])
    if hasattr(parsed_dates.dt, "tz") and parsed_dates.dt.tz is not None:
        parsed_dates = parsed_dates.dt.tz_localize(None)
    work_df["invoice_date"] = parsed_dates
    work_df["line_total"] = work_df["quantity"] * work_df["unit_price"]

    # Determine reference date
    if snapshot_date is None:
        snap_dt = default_snapshot_date(work_df)
    else:
        snap_dt = pd.to_datetime(snapshot_date)
        if hasattr(snap_dt, "tz") and snap_dt.tz is not None:
            snap_dt = snap_dt.tz_localize(None)
        max_dt = work_df["invoice_date"].max()
        if snap_dt < max_dt:
            raise ValueError(
                f"Snapshot date ({snap_dt.strftime('%Y-%m-%d')}) cannot be earlier than "
                f"the latest transaction date ({max_dt.strftime('%Y-%m-%d')})."
            )

    # Aggregations per customer
    agg_dict = {
        "invoice_date": ["max", "min"],
        "invoice_no": "nunique",
        "line_total": "sum",
    }
    has_country = "country" in work_df.columns
    if has_country:
        def _primary_mode(x):
            return x.mode()[0] if not x.empty and not x.mode().empty else "Unknown"
        agg_dict["country"] = _primary_mode

    grouped = work_df.groupby("customer_id").agg(agg_dict)

    # Flatten multi-index columns
    rfm = pd.DataFrame()
    rfm["customer_id"] = grouped.index.astype(str)
    
    last_dates = grouped[("invoice_date", "max")]
    first_dates = grouped[("invoice_date", "min")]
    
    # Recency in days
    rfm["recency"] = (snap_dt - last_dates).dt.days.values
    # Frequency (count of unique invoices, minimum 1)
    rfm["frequency"] = np.maximum(1, grouped[("invoice_no", "nunique")].values.astype(int))
    # Monetary (sum of line totals)
    rfm["monetary"] = np.round(grouped[("line_total", "sum")].values.astype(float), 2)
    
    rfm["first_purchase"] = first_dates.values
    rfm["last_purchase"] = last_dates.values

    if has_country:
        country_col = grouped["country"]
        rfm["country"] = country_col.iloc[:, 0].values if isinstance(country_col, pd.DataFrame) else country_col.values

    # Guarantee minimum recency of 0 days
    rfm["recency"] = rfm["recency"].clip(lower=0)

    # Sort by monetary descending
    rfm = rfm.sort_values(by="monetary", ascending=False).reset_index(drop=True)

    return rfm


def get_rfm_summary_stats(rfm_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Computes distribution statistics (mean, std, min, percentiles, max) for R, F, and M metrics.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        DataFrame produced by calculate_rfm().

    Returns
    -------
    dict
        Dictionary containing statistical summaries for 'recency', 'frequency', and 'monetary'.
    """
    stats = {}
    for metric in ["recency", "frequency", "monetary"]:
        if metric in rfm_df.columns:
            s = rfm_df[metric]
            stats[metric] = {
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2) if len(s) > 1 else 0.0,
                "min": round(float(s.min()), 2),
                "p25": round(float(s.quantile(0.25)), 2),
                "median": round(float(s.median()), 2),
                "p75": round(float(s.quantile(0.75)), 2),
                "max": round(float(s.max()), 2),
            }
    return stats
