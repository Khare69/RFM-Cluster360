"""
Unit tests for two-pass net-settlement transaction reconciliation.
"""

from datetime import datetime
import pandas as pd

from src.reconciler import reconcile_transactions


def test_reconcile_transactions_empty():
    df = pd.DataFrame(columns=["customer_id", "invoice_no", "invoice_date", "quantity", "unit_price"])
    reconciled_df, report = reconcile_transactions(df)
    assert reconciled_df.empty
    assert report.total_raw_records == 0
    assert report.net_settled_revenue == 0.0


def test_reconcile_transactions_full_refund():
    # Customer buys 2 items at $50 on Day 1, returns 2 items on Day 2
    data = {
        "customer_id": ["CUST1", "CUST1"],
        "invoice_no": ["INV001", "CINV001"],
        "invoice_date": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
        "quantity": [2.0, -2.0],
        "unit_price": [50.0, 50.0],
        "description": ["Widget A", "Widget A"],
    }
    df = pd.DataFrame(data)
    reconciled_df, report = reconcile_transactions(df)

    # Entire purchase was returned -> net quantity 0, should not be in settled_df
    assert reconciled_df.empty
    assert report.total_raw_records == 2
    assert report.matched_reversals == 1
    assert report.unmatched_reversals == 0
    assert report.gross_revenue == 100.0
    assert report.net_settled_revenue == 0.0


def test_reconcile_transactions_partial_refund():
    # Customer buys 5 items at $20, returns 2 items
    data = {
        "customer_id": ["CUST1", "CUST1"],
        "invoice_no": ["INV001", "CINV001"],
        "invoice_date": [datetime(2026, 1, 1), datetime(2026, 1, 5)],
        "quantity": [5.0, -2.0],
        "unit_price": [20.0, 20.0],
        "description": ["Item 1", "Item 1"],
    }
    df = pd.DataFrame(data)
    reconciled_df, report = reconcile_transactions(df)

    assert len(reconciled_df) == 1
    row = reconciled_df.iloc[0]
    assert row["net_quantity"] == 3.0
    assert row["net_amount"] == 60.0
    assert bool(row["is_reconciled"]) is True
    assert report.matched_reversals == 1
    assert report.gross_revenue == 100.0
    assert report.net_settled_revenue == 60.0


def test_reconcile_transactions_lookback_cutoff():
    # Return occurs outside lookback window (> 180 days)
    data = {
        "customer_id": ["CUST1", "CUST1"],
        "invoice_no": ["INV001", "CINV001"],
        "invoice_date": [datetime(2025, 1, 1), datetime(2025, 10, 1)],  # 273 days apart
        "quantity": [1.0, -1.0],
        "unit_price": [100.0, 100.0],
        "description": ["Item A", "Item A"],
    }
    df = pd.DataFrame(data)
    reconciled_df, report = reconcile_transactions(df, lookback_window_days=90)

    # Since return is outside the 90-day window, it is not matched against this purchase
    assert len(reconciled_df) == 1
    assert reconciled_df.iloc[0]["net_quantity"] == 1.0
    assert report.matched_reversals == 0
    assert report.unmatched_reversals == 1
