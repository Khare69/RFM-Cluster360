"""
Two-pass net-settlement transaction reconciler module.
Matches cancelled and refunded transactions against original parent purchases
to eliminate upward bias in customer Frequency and Monetary metrics.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class ReconciliationReport:
    """Structured report tracking net-settlement transaction reconciliation metrics."""
    total_raw_records: int
    purchase_records: int
    reversal_records: int
    matched_reversals: int
    unmatched_reversals: int
    gross_revenue: float
    net_settled_revenue: float
    reconciliation_rate_pct: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def reconcile_transactions(
    df: pd.DataFrame,
    lookback_window_days: int = 180,
    item_match_col: Optional[str] = "description",
) -> Tuple[pd.DataFrame, ReconciliationReport]:
    """
    Executes two-pass net-settlement reconciliation on raw transaction logs.

    Pass 1:
      Separates positive purchases from reversal/cancellation records.
    Pass 2:
      For each customer, matches reversal records against the most recent preceding
      purchase of the same item (or customer-level purchase if item column is unavailable)
      within `lookback_window_days`. Reconciles quantities and net amounts.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction DataFrame with columns ['customer_id', 'invoice_no', 'invoice_date', 'quantity', 'unit_price'].
    lookback_window_days : int, default 180
        Maximum time window in days between original purchase and return.
    item_match_col : str, optional
        Column name to match specific products (e.g., 'description' or 'stock_code').

    Returns
    -------
    Tuple[pd.DataFrame, ReconciliationReport]
        Reconciled DataFrame with columns ['net_quantity', 'net_amount', 'is_settled'] and ReconciliationReport.
    """
    required_cols = ["customer_id", "invoice_no", "invoice_date", "quantity", "unit_price"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing for reconciliation.")

    if df.empty:
        report = ReconciliationReport(
            total_raw_records=0,
            purchase_records=0,
            reversal_records=0,
            matched_reversals=0,
            unmatched_reversals=0,
            gross_revenue=0.0,
            net_settled_revenue=0.0,
            reconciliation_rate_pct=0.0,
        )
        empty_res = df.copy()
        empty_res["net_quantity"] = pd.Series(dtype=float)
        empty_res["net_amount"] = pd.Series(dtype=float)
        empty_res["is_reconciled"] = pd.Series(dtype=bool)
        return empty_res, report

    work_df = df.copy()
    work_df["invoice_date"] = pd.to_datetime(work_df["invoice_date"])
    work_df["quantity"] = pd.to_numeric(work_df["quantity"], errors="coerce").fillna(0.0)
    work_df["unit_price"] = pd.to_numeric(work_df["unit_price"], errors="coerce").fillna(0.0)
    work_df["raw_line_total"] = work_df["quantity"] * work_df["unit_price"]

    # Identify reversals: negative quantity or invoice prefix 'C'
    is_c_invoice = work_df["invoice_no"].astype(str).str.upper().str.startswith("C")
    is_reversal = (work_df["quantity"] < 0) | is_c_invoice

    purchases = work_df[~is_reversal].copy().sort_values("invoice_date")
    reversals = work_df[is_reversal].copy().sort_values("invoice_date")

    purchases["net_quantity"] = purchases["quantity"].astype(float)
    purchases["net_amount"] = purchases["raw_line_total"].astype(float)
    purchases["is_reconciled"] = False

    matched_reversals_count = 0
    unmatched_reversals_count = 0
    gross_revenue = float(purchases["raw_line_total"].sum()) if not purchases.empty else 0.0

    # Group purchases by customer for fast localized matching
    has_item = item_match_col is not None and item_match_col in work_df.columns
    lookback_delta = pd.Timedelta(days=lookback_window_days)

    if not reversals.empty and not purchases.empty:
        # Index purchases for mutable tracking during matching
        purchases_dict: Dict[str, List[int]] = {}
        for idx, row in purchases.iterrows():
            cid = str(row["customer_id"])
            if cid not in purchases_dict:
                purchases_dict[cid] = []
            purchases_dict[cid].append(idx)

        for _, rev_row in reversals.iterrows():
            cid = str(rev_row["customer_id"])
            rev_date = rev_row["invoice_date"]
            rev_qty = abs(float(rev_row["quantity"]))
            rev_item = str(rev_row[item_match_col]).strip().lower() if has_item else None

            matched = False
            candidate_indices = purchases_dict.get(cid, [])
            # Search in reverse chronological order (most recent purchase first)
            for p_idx in reversed(candidate_indices):
                p_date = purchases.at[p_idx, "invoice_date"]
                if p_date <= rev_date and (rev_date - p_date) <= lookback_delta:
                    # Item match check if available
                    if has_item:
                        p_item = str(purchases.at[p_idx, item_match_col]).strip().lower()
                        if rev_item and p_item != rev_item:
                            continue

                    available_qty = purchases.at[p_idx, "net_quantity"]
                    if available_qty > 0:
                        deduct_qty = min(available_qty, rev_qty)
                        unit_p = purchases.at[p_idx, "unit_price"]
                        purchases.at[p_idx, "net_quantity"] -= deduct_qty
                        purchases.at[p_idx, "net_amount"] -= (deduct_qty * unit_p)
                        purchases.at[p_idx, "is_reconciled"] = True
                        rev_qty -= deduct_qty
                        matched = True
                        if rev_qty <= 1e-6:
                            break

            if matched:
                matched_reversals_count += 1
            else:
                unmatched_reversals_count += 1
    else:
        unmatched_reversals_count = len(reversals)

    # Filter purchases with strictly positive net settled quantities
    settled_df = purchases[purchases["net_quantity"] > 0].copy()
    settled_df["net_amount"] = np.maximum(0.0, settled_df["net_amount"])
    net_settled_revenue = float(settled_df["net_amount"].sum()) if not settled_df.empty else 0.0

    total_rev_records = len(reversals)
    recon_rate = round((matched_reversals_count / total_rev_records * 100), 2) if total_rev_records > 0 else 100.0

    report = ReconciliationReport(
        total_raw_records=len(df),
        purchase_records=len(purchases),
        reversal_records=total_rev_records,
        matched_reversals=matched_reversals_count,
        unmatched_reversals=unmatched_reversals_count,
        gross_revenue=round(gross_revenue, 2),
        net_settled_revenue=round(net_settled_revenue, 2),
        reconciliation_rate_pct=recon_rate,
    )

    return settled_df.reset_index(drop=True), report
