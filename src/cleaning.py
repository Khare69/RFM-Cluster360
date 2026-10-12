"""
Data validation, cleaning, and transparent reporting module.
"""

from dataclasses import asdict, dataclass
import re
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class CleaningPolicy:
    """Configurable data cleaning and transaction handling policy."""
    policy_type: str = "gross"  # "gross" (drop returns/negatives) or "net" (retain refund transactions to net out spend)
    drop_cancelled_invoices: bool = True  # whether to drop invoices starting with 'C' in gross mode
    remove_duplicates: bool = False
    currency: str = "USD"
    return_time_window_days: Optional[int] = None


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
    rows_removed_missing_invoice: int = 0
    rows_removed_non_positive_qty: int = 0
    rows_removed_non_positive_price: int = 0
    rows_removed_invalid_date: int = 0
    duplicate_rows_detected: int = 0
    duplicates_removed: bool = False
    # Explicit Return & Refund Tracking (Phase 10)
    policy_type: str = "gross"
    gross_revenue: float = 0.0
    refund_amount: float = 0.0
    refund_count: int = 0
    refund_rate_pct: float = 0.0

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
    policy: Optional[CleaningPolicy] = None,
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
    policy : CleaningPolicy, optional
        Configurable cleaning policy (gross vs net returns, currency, cancellation handling).

    Returns
    -------
    Tuple[pd.DataFrame, CleaningReport]
        (Cleaned DataFrame with standardized column names, CleaningReport)
    """
    if policy is None:
        policy = CleaningPolicy(remove_duplicates=remove_duplicates)
    elif remove_duplicates:
        policy.remove_duplicates = True
    actual_remove_duplicates = policy.remove_duplicates
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
        | (work_df["customer_id_str"].isin(["", "nan", "none", "null"]))
    )
    rows_missing_customer = int(missing_customer_mask.sum())
    work_df = work_df[~missing_customer_mask].drop(columns=["customer_id_str"])

    # Ensure customer_id is a clean string/identifier (normalize float representations like 12345.0 to 12345
    # while strictly preserving leading zeros like "00123" and distinct formats like "1E5")
    def _clean_id(val):
        if val is None or pd.isna(val):
            return ""
        if isinstance(val, (float, np.floating)):
            if val.is_integer():
                return str(int(val))
            return str(val).strip()
        s = str(val).strip()
        # Only strip trailing decimal .0 suffixes to preserve original string identifiers
        return re.sub(r"\.0+$", "", s)

    work_df["customer_id"] = work_df["customer_id"].apply(_clean_id)

    # Step 3: Parse and validate numeric columns (safely handle currency signs, commas, accounting parens, whitespace)
    def _to_clean_numeric(series):
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        cleaned = series.astype(str).str.strip()
        # Convert accounting parenthesis format: (10.50) -> -10.50, ($10.50) -> -$10.50
        cleaned = cleaned.str.replace(r"^\s*\(\s*(.*?)\s*\)\s*$", r"-\1", regex=True)
        cleaned = cleaned.str.replace(r"[^\d.-]", "", regex=True)
        return pd.to_numeric(cleaned, errors="coerce")

    work_df["quantity"] = _to_clean_numeric(work_df["quantity"])
    work_df["unit_price"] = _to_clean_numeric(work_df["unit_price"])

    # Step 4: Handle cancelled transactions
    work_df["invoice_no_str"] = work_df["invoice_no"].astype(str).str.strip()
    is_c_invoice = work_df["invoice_no_str"].str.upper().str.startswith("C")

    if policy.policy_type == "net":
        # In net mode, keep C-invoices that have negative quantity as legitimate return transactions
        is_cancelled_non_return = is_c_invoice & (work_df["quantity"] >= 0)
        rows_cancelled = int(is_cancelled_non_return.sum())
        work_df = work_df[~is_cancelled_non_return]
    else:
        # In gross mode, discard all C-invoices as cancellations
        rows_cancelled = int(is_c_invoice.sum())
        work_df = work_df[~is_c_invoice]

    # Step 4b: Handle null, empty, or placeholder invoice_no (case-insensitive)
    missing_invoice_mask = (
        work_df["invoice_no"].isna()
        | (work_df["invoice_no_str"].str.lower().isin(["", "nan", "none", "null", "0"]))
    )
    rows_missing_invoice = int(missing_invoice_mask.sum())
    work_df = work_df[~missing_invoice_mask].drop(columns=["invoice_no_str"])

    # Step 5: Handle non-positive quantities (returns vs zero/invalid)
    if policy.policy_type == "net":
        # Retain negative quantities as refunds; drop only NaN or 0 quantities
        zero_or_nan_qty = work_df["quantity"].isna() | (work_df["quantity"] == 0)
        rows_non_pos_qty = int(zero_or_nan_qty.sum())
        work_df = work_df[~zero_or_nan_qty]
    else:
        # Gross mode: drop all non-positive quantities
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
    if actual_remove_duplicates and duplicate_count > 0:
        work_df = work_df.drop_duplicates()

    # Compute final metrics & revenue breakdown
    total_rows_after = len(work_df)
    total_customers = work_df["customer_id"].nunique() if total_rows_after > 0 else 0

    if total_rows_after > 0:
        line_totals = work_df["quantity"] * work_df["unit_price"]
        pos_mask = line_totals > 0
        neg_mask = line_totals < 0
        work_df["is_refund"] = neg_mask | work_df["invoice_no"].astype(str).str.upper().str.startswith("C")
        pos_revenue = float(line_totals[pos_mask].sum()) if pos_mask.any() else 0.0
        refund_amount = float(abs(line_totals[neg_mask].sum())) if neg_mask.any() else 0.0
        refund_count = int(neg_mask.sum())
        refund_rate_pct = round((refund_amount / pos_revenue * 100), 2) if pos_revenue > 0 else 0.0
        total_revenue = (pos_revenue - refund_amount) if policy.policy_type == "net" else pos_revenue
    else:
        work_df["is_refund"] = pd.Series(dtype=bool)
        pos_revenue = 0.0
        refund_amount = 0.0
        refund_count = 0
        refund_rate_pct = 0.0
        total_revenue = 0.0

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
        rows_removed_missing_invoice=rows_missing_invoice,
        rows_removed_non_positive_qty=rows_non_pos_qty,
        rows_removed_non_positive_price=rows_non_pos_price,
        rows_removed_invalid_date=rows_invalid_date,
        duplicate_rows_detected=duplicate_count,
        duplicates_removed=actual_remove_duplicates,
        policy_type=policy.policy_type,
        gross_revenue=round(pos_revenue, 2),
        refund_amount=round(refund_amount, 2),
        refund_count=refund_count,
        refund_rate_pct=refund_rate_pct,
    )

    # Sort by invoice_date
    work_df = work_df.sort_values("invoice_date").reset_index(drop=True)

    return work_df, report

