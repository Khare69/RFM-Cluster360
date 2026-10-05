"""
Data validation, cleaning, and transparent reporting module.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class CleaningReport:
    """Structured report tracking every transformation and removed row count."""
    total_rows_before: int
    total_rows_after: int
    total_customers: int
    total_revenue: float
    date_min: Optional[str]
    date_max: Optional[str]
    rows_removed_missing_customer: int = 0
    rows_removed_cancelled: int = 0
    rows_removed_non_positive_qty: int = 0
    rows_removed_non_positive_price: int = 0
    rows_removed_invalid_date: int = 0
    duplicate_rows_detected: int = 0
    duplicates_removed: bool = False

    @property
    def total_rows_removed(self) -> int:
        return self.total_rows_before - self.total_rows_after

    @property
    def retention_percentage(self) -> float:
        if self.total_rows_before == 0:
            return 0.0
        return round((self.total_rows_after / self.total_rows_before) * 100, 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_rows_removed"] = self.total_rows_removed
        d["retention_percentage"] = self.retention_percentage
        return d


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Returns exact duplicate rows across all columns."""
    return df[df.duplicated(keep=False)]


def clean_transactions(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    remove_duplicates: bool = False,
) -> Tuple[pd.DataFrame, CleaningReport]:
    """
    Cleans raw transaction DataFrame according to business rules and produces a transparent report.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction DataFrame.
    mapping : dict
        Mapping of canonical name -> actual column name in df.
    remove_duplicates : bool, default False
        Whether to drop exact duplicate rows.

    Returns
    -------
    Tuple[pd.DataFrame, CleaningReport]
        (Cleaned DataFrame with standardized column names, CleaningReport)
    """
    total_rows_before = len(df)
    work_df = df.copy()

    # Invert mapping to select and rename columns
    inv_map = {actual: canonical for canonical, actual in mapping.items() if actual is not None}
    
    # Rename mapped columns
    work_df = work_df.rename(columns=inv_map)

    # Keep only canonical columns + whatever optional ones were mapped
    canonical_present = [col for col in ["customer_id", "invoice_no", "invoice_date", "quantity", "unit_price", "country", "description"] if col in work_df.columns]
    work_df = work_df[canonical_present].copy()

    # Step 1: Detect duplicates before removal
    duplicates_df = detect_duplicates(work_df)
    duplicate_count = len(duplicates_df)

    # Step 2: Handle missing customer IDs (null, NaN, empty string, "nan", "none", etc.)
    work_df["customer_id_str"] = work_df["customer_id"].astype(str).str.strip().str.lower()
    missing_customer_mask = (
        work_df["customer_id"].isna()
        | (work_df["customer_id_str"].isin(["", "nan", "none", "null", "0"]))
    )
    rows_missing_customer = int(missing_customer_mask.sum())
    work_df = work_df[~missing_customer_mask].drop(columns=["customer_id_str"])

    # Ensure customer_id is a clean string/identifier (e.g. convert float 12345.0 to "12345")
    def _clean_id(val):
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
        return str(val).strip()

    work_df["customer_id"] = work_df["customer_id"].apply(_clean_id)

    # Step 3: Parse and validate numeric columns (safely handle currency signs, commas, whitespace)
    def _to_clean_numeric(series):
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        cleaned = series.astype(str).str.replace(r"[^\d.-]", "", regex=True)
        return pd.to_numeric(cleaned, errors="coerce")

    work_df["quantity"] = _to_clean_numeric(work_df["quantity"])
    work_df["unit_price"] = _to_clean_numeric(work_df["unit_price"])

    # Step 4: Handle cancelled transactions (invoice_no starting with 'C' or 'c')
    work_df["invoice_no_str"] = work_df["invoice_no"].astype(str).str.strip()
    cancelled_mask = work_df["invoice_no_str"].str.upper().str.startswith("C")
    rows_cancelled = int(cancelled_mask.sum())
    work_df = work_df[~cancelled_mask].drop(columns=["invoice_no_str"])

    # Step 5: Remove non-positive quantities (returns/adjustments/zeros)
    non_pos_qty_mask = work_df["quantity"].isna() | (work_df["quantity"] <= 0)
    rows_non_pos_qty = int(non_pos_qty_mask.sum())
    work_df = work_df[~non_pos_qty_mask]

    # Step 6: Remove non-positive unit prices (freebies, damages, data errors)
    non_pos_price_mask = work_df["unit_price"].isna() | (work_df["unit_price"] <= 0)
    rows_non_pos_price = int(non_pos_price_mask.sum())
    work_df = work_df[~non_pos_price_mask]

    # Step 7: Parse dates
    work_df["invoice_date"] = pd.to_datetime(work_df["invoice_date"], errors="coerce")
    invalid_date_mask = work_df["invoice_date"].isna()
    rows_invalid_date = int(invalid_date_mask.sum())
    work_df = work_df[~invalid_date_mask]

    # Step 8: Optional duplicate removal
    if remove_duplicates and duplicate_count > 0:
        work_df = work_df.drop_duplicates()

    # Compute final metrics
    total_rows_after = len(work_df)
    total_customers = work_df["customer_id"].nunique() if total_rows_after > 0 else 0
    total_revenue = float((work_df["quantity"] * work_df["unit_price"]).sum()) if total_rows_after > 0 else 0.0

    date_min = work_df["invoice_date"].min().strftime("%Y-%m-%d") if total_rows_after > 0 else None
    date_max = work_df["invoice_date"].max().strftime("%Y-%m-%d") if total_rows_after > 0 else None

    report = CleaningReport(
        total_rows_before=total_rows_before,
        total_rows_after=total_rows_after,
        total_customers=total_customers,
        total_revenue=round(total_revenue, 2),
        date_min=date_min,
        date_max=date_max,
        rows_removed_missing_customer=rows_missing_customer,
        rows_removed_cancelled=rows_cancelled,
        rows_removed_non_positive_qty=rows_non_pos_qty,
        rows_removed_non_positive_price=rows_non_pos_price,
        rows_removed_invalid_date=rows_invalid_date,
        duplicate_rows_detected=duplicate_count,
        duplicates_removed=remove_duplicates,
    )

    # Sort by invoice_date
    work_df = work_df.sort_values("invoice_date").reset_index(drop=True)

    return work_df, report
