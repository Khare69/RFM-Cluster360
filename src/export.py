"""
Data export utility module.
Prepares clean, labeled customer segmentation DataFrames and serializes them to CSV bytes for download.
"""

import io
from typing import List, Optional
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
