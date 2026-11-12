"""
Unit and integration tests for Out-of-Time Backtesting and Segment Migration.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.backtesting import (
    compute_segment_transition_matrix,
    compute_windowed_rfm,
    evaluate_forward_monetary_realization,
    get_customer_migration_details,
    run_backtesting_pipeline,
    split_transaction_windows,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_temporal_transactions() -> pd.DataFrame:
    """Creates a 6-month synthetic transaction log with repeat and churned customers."""
    records = []
    # Customer 101: Active in both T1 and T2 (Champion)
    records.append({"customer_id": "101", "invoice_no": "INV001", "invoice_date": "2023-01-15", "quantity": 5, "unit_price": 20.0})
    records.append({"customer_id": "101", "invoice_no": "INV002", "invoice_date": "2023-02-10", "quantity": 4, "unit_price": 25.0})
    records.append({"customer_id": "101", "invoice_no": "INV003", "invoice_date": "2023-04-15", "quantity": 6, "unit_price": 30.0})

    # Customer 102: Active in T1 only (Churned)
    records.append({"customer_id": "102", "invoice_no": "INV004", "invoice_date": "2023-01-20", "quantity": 2, "unit_price": 50.0})
    records.append({"customer_id": "102", "invoice_no": "INV005", "invoice_date": "2023-02-28", "quantity": 1, "unit_price": 40.0})

    # Customer 103: Active in both T1 and T2 (Loyal)
    records.append({"customer_id": "103", "invoice_no": "INV006", "invoice_date": "2023-02-15", "quantity": 3, "unit_price": 15.0})
    records.append({"customer_id": "103", "invoice_no": "INV007", "invoice_date": "2023-05-10", "quantity": 4, "unit_price": 20.0})

    # Customer 104: Newly acquired in T2 (New Customer)
    records.append({"customer_id": "104", "invoice_no": "INV008", "invoice_date": "2023-04-01", "quantity": 2, "unit_price": 35.0})
    records.append({"customer_id": "104", "invoice_no": "INV009", "invoice_date": "2023-05-20", "quantity": 3, "unit_price": 40.0})

    # Extra customers to satisfy clustering minimums
    for cid in range(200, 220):
        records.append({
            "customer_id": str(cid),
            "invoice_no": f"INV{cid}",
            "invoice_date": "2023-01-10" if cid % 2 == 0 else "2023-05-15",
            "quantity": (cid % 5) + 1,
            "unit_price": 10.0 + (cid % 10),
        })

    df = pd.DataFrame(records)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def test_split_transaction_windows_valid(sample_temporal_transactions):
    """Verify clean partition into T1 (<= cutoff) and T2 (> cutoff) with no leakage."""
    cutoff = "2023-03-15"
    t1_df, t2_df = split_transaction_windows(sample_temporal_transactions, cutoff_date=cutoff)

    assert not t1_df.empty, "T1 window must not be empty"
    assert not t2_df.empty, "T2 window must not be empty"
    assert len(t1_df) + len(t2_df) == len(sample_temporal_transactions)

    assert (t1_df["invoice_date"] <= pd.Timestamp(cutoff)).all(), "T1 must only contain records <= cutoff"
    assert (t2_df["invoice_date"] > pd.Timestamp(cutoff)).all(), "T2 must only contain records > cutoff"


def test_split_transaction_windows_invalid_dates(sample_temporal_transactions):
    """Verify ValueError is raised if cutoff is out of bounds or data is empty."""
    with pytest.raises(ValueError, match="must fall strictly within dataset date range"):
        split_transaction_windows(sample_temporal_transactions, cutoff_date="2020-01-01")

    with pytest.raises(ValueError, match="must fall strictly within dataset date range"):
        split_transaction_windows(sample_temporal_transactions, cutoff_date="2025-01-01")

    with pytest.raises(ValueError, match="DataFrame is empty"):
        split_transaction_windows(pd.DataFrame(), cutoff_date="2023-03-01")


def test_compute_windowed_rfm(sample_temporal_transactions):
    """Verify RFM calculation independently across T1 and T2 windows."""
    cutoff = "2023-03-15"
    t1_rfm, t2_rfm = compute_windowed_rfm(sample_temporal_transactions, cutoff_date=cutoff)

    assert "customer_id" in t1_rfm.columns and "customer_id" in t2_rfm.columns
    assert "recency" in t1_rfm.columns and "frequency" in t1_rfm.columns and "monetary" in t1_rfm.columns
    assert (t1_rfm["monetary"] > 0).all()
    assert (t2_rfm["monetary"] > 0).all()


def test_compute_segment_transition_matrix():
    """Verify transition counts and row-normalized Markov probabilities."""
    t1_labeled = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4"],
        "segment": ["Champions", "Champions", "Loyal", "At Risk"],
    })
    t2_labeled = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C5"],
        "segment": ["Champions", "Loyal", "At Risk", "Champions"],  # C4 is churned, C5 is new
    })

    res = compute_segment_transition_matrix(t1_labeled, t2_labeled)

    prob_matrix = res["prob_matrix"]
    counts_matrix = res["counts_matrix"]
    assert counts_matrix is not None
    assert counts_matrix.loc["Champions", "Champions"] == 1

    # Verify probability normalization: all row sums must be 1.0
    for seg, row in prob_matrix.iterrows():
        assert np.isclose(row.sum(), 1.0, atol=1e-5), f"Row sum for segment {seg} must equal 1.0"

    # Champions: C1 stayed Champions, C2 moved to Loyal -> 50% Champions, 50% Loyal
    assert prob_matrix.loc["Champions", "Champions"] == 0.5
    assert prob_matrix.loc["Champions", "Loyal"] == 0.5

    # At Risk: C4 churned -> 100% Churned (Inactive)
    assert prob_matrix.loc["At Risk", "Churned (Inactive)"] == 1.0

    # Summary KPIs
    assert res["total_t1_customers"] == 4
    assert res["retained_count"] == 3  # C1, C2, C3
    assert res["churned_count"] == 1   # C4
    assert res["new_customers_count"] == 1  # C5
    assert res["retention_rate"] == 0.75
    assert res["churn_rate"] == 0.25


def test_evaluate_forward_monetary_realization():
    """Verify forward spend and order counts aggregated by T1 persona."""
    t1_labeled = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "segment": ["Champions", "Champions", "Lost"],
    })
    t2_rfm = pd.DataFrame({
        "customer_id": ["C1", "C3"],
        "frequency": [3, 1],
        "monetary": [300.0, 50.0],
    })

    summary = evaluate_forward_monetary_realization(t1_labeled, t2_rfm)

    assert "segment" in summary.columns
    assert "total_forward_spend" in summary.columns
    assert "forward_retention_pct" in summary.columns

    # Champions had 2 customers; C1 spent 300, C2 spent 0 -> total 300, retention 50%
    champ_row = summary[summary["segment"] == "Champions"].iloc[0]
    assert champ_row["customer_count"] == 2
    assert champ_row["forward_active_count"] == 1
    assert champ_row["total_forward_spend"] == 300.0
    assert champ_row["forward_retention_pct"] == 50.0
    assert champ_row["avg_forward_spend"] == 150.0


def test_get_customer_migration_details():
    """Verify customer-level migration drilldown classification."""
    t1_labeled = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "segment": ["Champions", "Champions", "Loyal"],
        "monetary": [100.0, 200.0, 50.0],
    })
    t2_labeled = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "segment": ["Champions", "At Risk"],
        "monetary": [150.0, 80.0],
    })

    details = get_customer_migration_details(t1_labeled, t2_labeled)

    c1_row = details[details["customer_id"] == "C1"].iloc[0]
    assert c1_row["migration_status"] == "Retained (Stable)"
    assert c1_row["t2_spend"] == 150.0

    c2_row = details[details["customer_id"] == "C2"].iloc[0]
    assert c2_row["migration_status"] == "Migrated"

    c3_row = details[details["customer_id"] == "C3"].iloc[0]
    assert c3_row["migration_status"] == "Churned"
    assert c3_row["t2_spend"] == 0.0


def test_run_backtesting_pipeline_end_to_end(sample_temporal_transactions):
    """Verify full end-to-end backtesting pipeline execution on sample data."""
    results = run_backtesting_pipeline(
        df=sample_temporal_transactions,
        cutoff_date="2023-03-15",
        k=3,
        random_state=42,
    )

    assert "prob_matrix" in results
    assert "counts_matrix" in results
    assert "forward_metrics" in results
    assert "retention_rate" in results
    assert 0.0 <= results["retention_rate"] <= 1.0
    assert 0.0 <= results["churn_rate"] <= 1.0


def test_page_7_backtesting_apptest_smoke(sample_temporal_transactions):
    """Verify Page 7 renders without unhandled exceptions under both uninitialized and populated states."""
    page_path = REPO_ROOT / "pages" / "7_Segment_Backtesting.py"

    # 1. Uninitialized state
    at_empty = AppTest.from_file(str(page_path))
    at_empty.run()
    assert not at_empty.exception, f"Page 7 empty state raised exception: {at_empty.exception}"
    assert len(at_empty.warning) > 0, "Page 7 must display a warning when session state is uninitialized"

    # 2. Populated state
    at_populated = AppTest.from_file(str(page_path))
    at_populated.session_state["cleaned_df"] = sample_temporal_transactions
    at_populated.session_state["currency_symbol"] = "$"
    at_populated.session_state["k"] = 3
    at_populated.run()

    assert not at_populated.exception, f"Page 7 populated state raised exception: {at_populated.exception}"
    assert len(at_populated.metric) >= 4, "Page 7 must display 4 summary KPI metric cards"
