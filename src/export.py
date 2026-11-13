"""
Data export utility module.
Prepares clean, labeled customer segmentation DataFrames and serializes them to CSV bytes for download.
"""

import hashlib
import io
from typing import List, Optional
import numpy as np
import pandas as pd


def sanitize_formula_injection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitizes string and object columns against CSV formula injection (CSV injection / DDE).
    Prepends a single quote "'" to cells starting with '=', '+', '-', '@', '\t', or '\r'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to sanitize.

    Returns
    -------
    pd.DataFrame
        Sanitized DataFrame with escaped formula trigger characters.
    """
    sanitized = df.copy()
    str_cols = sanitized.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        sanitized[col] = sanitized[col].apply(
            lambda x: f"'{x}" if isinstance(x, str) and len(x) > 0 and x[0] in ("=", "+", "-", "@", "\t", "\r") else x
        )
    return sanitized


def prepare_export_dataframe(
    labeled_df: pd.DataFrame,
    include_dates: bool = True,
    currency_symbol: str = "$",
    sanitize_formulas: bool = True,
) -> pd.DataFrame:
    """
    Standardizes column names and formats for external export.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        DataFrame containing RFM metrics, cluster ID, and segment label.
    include_dates : bool, default True
        Whether to keep first/last purchase timestamps.
    currency_symbol : str, default "$"
        Currency symbol to embed in the monetary header.
    sanitize_formulas : bool, default True
        Whether to sanitize formula triggers in string columns against CSV injection.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame ready for CSV export.
    """
    export_df = labeled_df.copy()

    # Column rename mapping for human-readable headers
    rename_map = {
        "customer_id": "Customer ID",
        "recency": "Recency (Days)",
        "frequency": "Frequency (Orders)",
        "monetary": f"Monetary ({currency_symbol})",
        "cluster": "Cluster ID",
        "segment": "Customer Segment",
        "first_purchase": "First Purchase Date",
        "last_purchase": "Last Purchase Date",
        "country": "Country",
    }

    # Format dates if present
    if "first_purchase" in export_df.columns:
        export_df["first_purchase"] = pd.to_datetime(export_df["first_purchase"]).dt.strftime("%Y-%m-%d")
    if "last_purchase" in export_df.columns:
        export_df["last_purchase"] = pd.to_datetime(export_df["last_purchase"]).dt.strftime("%Y-%m-%d")

    cols_to_keep = [c for c in rename_map.keys() if c in export_df.columns]
    if not include_dates:
        cols_to_keep = [c for c in cols_to_keep if c not in ["first_purchase", "last_purchase"]]

    export_df = export_df[cols_to_keep].rename(columns=rename_map)
    if sanitize_formulas:
        export_df = sanitize_formula_injection(export_df)
    return export_df


def to_csv_bytes(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    sanitize_formulas: bool = True,
) -> bytes:
    """
    Serializes a DataFrame into UTF-8 encoded CSV bytes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export.
    columns : list of str, optional
        Specific column subset to include.
    sanitize_formulas : bool, default True
        Whether to escape potential formula injection triggers in string cells.

    Returns
    -------
    bytes
        UTF-8 encoded CSV string as bytes.
    """
    export_df = df[columns] if columns is not None else df
    if sanitize_formulas:
        export_df = sanitize_formula_injection(export_df)
    return export_df.to_csv(index=False).encode("utf-8")


def to_parquet_bytes(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> bytes:
    """
    Serializes a DataFrame into Apache Parquet binary bytes using pyarrow.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export.
    columns : list of str, optional
        Specific column subset to include.

    Returns
    -------
    bytes
        Apache Parquet file content as bytes.
    """
    export_df = df[columns] if columns is not None else df
    buffer = io.BytesIO()
    export_df.to_parquet(buffer, engine="pyarrow", index=False)
    return buffer.getvalue()


def prepare_crm_reverse_etl_payload(
    labeled_df: pd.DataFrame,
    campaign_salt: str = "rfm_cluster360_campaign_v1",
    holdout_percentage: int = 10,
) -> pd.DataFrame:
    """
    Formats customer segmentation and CLV metrics into standardized Reverse-ETL
    CRM schemas (compatible with Klaviyo, Bloomreach, Google Value-Based Bidding).

    Applies deterministic hashing to assign a stratified control holdout flag (default 10%),
    allowing marketing teams to measure the true causal lift of segment campaigns.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        DataFrame containing customer segmentation results.
    campaign_salt : str, default 'rfm_cluster360_campaign_v1'
        Salt string for deterministic holdout hashing.
    holdout_percentage : int, default 10
        Percentage of audience to allocate to control holdout.

    Returns
    -------
    pd.DataFrame
        Clean, validated CRM-ready schema.
    """
    df = labeled_df.copy()

    crm_df = pd.DataFrame()
    crm_df["customer_id"] = df["customer_id"].astype(str)
    crm_df["recency_days"] = np.round(df["recency"]).astype(int) if "recency" in df.columns else 0
    crm_df["frequency_orders"] = np.round(df["frequency"]).astype(int) if "frequency" in df.columns else 1
    crm_df["monetary_spend"] = np.round(df["monetary"].astype(float), 2) if "monetary" in df.columns else 0.0

    if "segment" in df.columns:
        crm_df["segment_name"] = df["segment"].astype(str)
    elif "Customer Segment" in df.columns:
        crm_df["segment_name"] = df["Customer Segment"].astype(str)
    else:
        crm_df["segment_name"] = "Unassigned"

    if "cluster" in df.columns:
        crm_df["cluster_id"] = df["cluster"].astype(int)
    elif "Cluster ID" in df.columns:
        crm_df["cluster_id"] = df["Cluster ID"].astype(int)
    else:
        crm_df["cluster_id"] = 0

    if "predicted_clv_6m" in df.columns:
        crm_df["predicted_clv_6m"] = np.round(df["predicted_clv_6m"].astype(float), 2)
    else:
        crm_df["predicted_clv_6m"] = 0.0

    if "churn_probability" in df.columns:
        crm_df["churn_risk_pct"] = np.round(df["churn_probability"].astype(float) * 100, 1)
    elif "p_alive" in df.columns:
        crm_df["churn_risk_pct"] = np.round((1.0 - df["p_alive"].astype(float)) * 100, 1)
    else:
        crm_df["churn_risk_pct"] = 0.0

    # Deterministic control holdout hashing
    def _is_holdout(cid: str) -> bool:
        key = f"{cid}_{campaign_salt}".encode("utf-8")
        h = int(hashlib.sha256(key).hexdigest(), 16)
        return (h % 100) < holdout_percentage

    crm_df["control_holdout"] = crm_df["customer_id"].apply(_is_holdout)

    return crm_df
