"""
Unit tests for RFM (Recency, Frequency, Monetary) metric calculations.
"""

from datetime import datetime
import pandas as pd
import pytest

from src.rfm import calculate_rfm, default_snapshot_date, get_rfm_summary_stats


def test_calculate_rfm_hand_calculated(clean_test_transactions):
    # Snapshot date: 2023-03-16 (1 day after Customer A's latest purchase)
    snapshot = "2023-03-16"
    rfm = calculate_rfm(clean_test_transactions, snapshot_date=snapshot)

    assert len(rfm) == 3
    assert "customer_id" in rfm.columns
    assert "recency" in rfm.columns
    assert "frequency" in rfm.columns
    assert "monetary" in rfm.columns

    rfm_map = rfm.set_index("customer_id")

    # Customer A:
    # Latest purchase: 2023-03-15 -> Recency = 1 day
    # Unique orders: 101, 102 -> Frequency = 2
    # Monetary: (2*10) + (1*5) + (3*20) = $85.0
    cust_a = rfm_map.loc["A"]
    assert cust_a["recency"] == 1
    assert cust_a["frequency"] == 2
    assert cust_a["monetary"] == 85.0

    # Customer B:
    # Latest purchase: 2023-01-05 -> Recency = 70 days (Jan 5 to Mar 16)
    # Unique orders: 103 -> Frequency = 1
    # Monetary: (1*100) = $100.0
    cust_b = rfm_map.loc["B"]
    assert cust_b["recency"] == 70
    assert cust_b["frequency"] == 1
    assert cust_b["monetary"] == 100.0

    # Customer C:
    # Latest purchase: 2023-03-10 -> Recency = 6 days (Mar 10 to Mar 16)
    # Unique orders: 104, 105, 106 -> Frequency = 3
    # Monetary: (5*20) + (5*20) + (5*20) = $300.0
    cust_c = rfm_map.loc["C"]
    assert cust_c["recency"] == 6
    assert cust_c["frequency"] == 3
    assert cust_c["monetary"] == 300.0


def test_default_snapshot_date(clean_test_transactions):
    snap = default_snapshot_date(clean_test_transactions)
    # Max date is 2023-03-15, default snapshot should be 2023-03-16
    assert snap.strftime("%Y-%m-%d") == "2023-03-16"


def test_calculate_rfm_snapshot_in_past_error(clean_test_transactions):
    # Snapshot earlier than dataset transactions should raise ValueError
    with pytest.raises(ValueError, match="cannot be earlier"):
        calculate_rfm(clean_test_transactions, snapshot_date="2022-01-01")


def test_get_rfm_summary_stats(clean_test_transactions):
    rfm = calculate_rfm(clean_test_transactions)
    stats = get_rfm_summary_stats(rfm)
    assert "recency" in stats
    assert "frequency" in stats
    assert "monetary" in stats
    assert stats["monetary"]["max"] == 300.0
    assert stats["monetary"]["min"] == 85.0
