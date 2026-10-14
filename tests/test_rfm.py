"""
Unit tests for RFM (Recency, Frequency, Monetary) metric calculations.
"""

import pandas as pd
import pytest

from src.rfm import (
    calculate_rfm,
    compute_rfm_quintiles,
    default_snapshot_date,
    get_rfm_summary_stats,
)


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


def test_rfm_summary_scatter_zero_monetary():
    """Verify rfm_summary_scatter handles $0.00 monetary values safely without log_y -inf crash."""
    from viz.overview_charts import rfm_summary_scatter

    df = pd.DataFrame([
        {"customer_id": "C1", "recency": 10, "frequency": 1, "monetary": 0.0, "segment": "Champions"},
        {"customer_id": "C2", "recency": 5, "frequency": 2, "monetary": 100.0, "segment": "Loyal Customers"},
    ])
    fig = rfm_summary_scatter(df)
    assert fig is not None
    assert len(fig.data) > 0


def test_calculate_rfm_defensive_invoice_filtering():
    """Verify calculate_rfm defensively ignores placeholder invoice values ('None', 'NULL', '0')."""
    df = pd.DataFrame([
        {"customer_id": "C1", "invoice_no": "101", "invoice_date": "2023-01-01", "quantity": 1, "unit_price": 50.0},
        {"customer_id": "C1", "invoice_no": "None", "invoice_date": "2023-01-02", "quantity": 2, "unit_price": 50.0},
        {"customer_id": "C1", "invoice_no": "0", "invoice_date": "2023-01-03", "quantity": 1, "unit_price": 50.0},
    ])
    rfm = calculate_rfm(df, snapshot_date="2023-01-10")
    assert len(rfm) == 1
    # Only valid order 101 should count
    assert rfm.iloc[0]["frequency"] == 1
    assert rfm.iloc[0]["monetary"] == 50.0


def test_silhouette_plot_negative_scores():
    """Verify silhouette_plot adjusts y-axis range properly when scores include negative values."""
    from viz.rfm_charts import silhouette_plot

    eval_results = {
        "k_values": [2, 3, 4],
        "silhouette_scores": [-0.15, -0.05, 0.20],
        "best_k": 4,
    }
    fig = silhouette_plot(eval_results)
    assert fig is not None
    assert fig.layout.yaxis.range[0] < 0.0


def test_calculate_rfm_with_net_refunds_policy():
    """Verify RFM calculation nets refund monetary amounts, counts only positive purchase orders, and preserves true purchase recency."""
    df = pd.DataFrame([
        # Purchase on 2023-01-01
        {"customer_id": "C1", "invoice_no": "101", "invoice_date": "2023-01-01", "quantity": 2, "unit_price": 50.0, "is_refund": False},
        # Refund on 2023-01-10
        {"customer_id": "C1", "invoice_no": "C102", "invoice_date": "2023-01-10", "quantity": -1, "unit_price": 30.0, "is_refund": True},
        # Customer C2 with return exceeding purchase (negative net spend bounded to 0.0)
        {"customer_id": "C2", "invoice_no": "103", "invoice_date": "2023-01-05", "quantity": 1, "unit_price": 20.0, "is_refund": False},
        {"customer_id": "C2", "invoice_no": "C104", "invoice_date": "2023-01-06", "quantity": -1, "unit_price": 50.0, "is_refund": True},
    ])
    rfm = calculate_rfm(df, snapshot_date="2023-01-15")
    rfm_map = rfm.set_index("customer_id")

    # Customer C1:
    # Purchase date 2023-01-01 -> Recency = 14 days (NOT 5 days from return!)
    # Orders = 1 (only purchase 101, NOT C102)
    # Monetary = 100 - 30 = 70.0
    c1 = rfm_map.loc["C1"]
    assert c1["recency"] == 14
    assert c1["frequency"] == 1
    assert c1["monetary"] == 70.0

    # Customer C2:
    # Monetary net is 20 - 50 = -30 -> bounded at 0.0
    c2 = rfm_map.loc["C2"]
    assert c2["recency"] == 10
    assert c2["frequency"] == 1
    assert c2["monetary"] == 0.0


def test_rfm_histograms_custom_currency():
    """Verify rfm_histograms properly formats axis and subplot titles with selected currency symbol."""
    from viz.rfm_charts import rfm_histograms

    df = pd.DataFrame([
        {"customer_id": "C1", "recency": 10, "frequency": 2, "monetary": 150.0},
        {"customer_id": "C2", "recency": 20, "frequency": 3, "monetary": 300.0},
    ])
    fig = rfm_histograms(df, currency_symbol="€")
    assert fig is not None
    assert "€ Total Spend" in fig.layout.annotations[2].text


def test_cluster_scatter_custom_currency():
    """Verify 3D and 2D cluster scatter projections format axes and hover templates with custom currency."""
    from viz.cluster_scatter import scatter_2d_projections, scatter_3d_clusters

    df = pd.DataFrame([
        {"customer_id": "C1", "recency": 10, "frequency": 2, "monetary": 150.0, "segment": "Champions"},
        {"customer_id": "C2", "recency": 30, "frequency": 1, "monetary": 50.0, "segment": "At Risk"},
    ])
    fig_3d = scatter_3d_clusters(df, currency_symbol="₹")
    assert fig_3d is not None

    fig_2d = scatter_2d_projections(df, currency_symbol="₹")
    assert fig_2d is not None
    assert "Monetary (₹ Log)" in fig_2d.layout.yaxis2.title.text


def test_customer_timeline_and_benchmark_custom_currency():
    """Verify customer purchase timeline and benchmark charts reflect custom currency symbol."""
    from viz.customer_timeline import customer_benchmark_bar, customer_purchase_timeline

    rfm_df = pd.DataFrame([
        {"customer_id": "C1", "recency": 10, "frequency": 2, "monetary": 150.0},
        {"customer_id": "C2", "recency": 30, "frequency": 1, "monetary": 50.0},
    ])
    tx_df = pd.DataFrame([
        {"invoice_no": "101", "invoice_date": pd.to_datetime("2023-01-01"), "quantity": 2, "unit_price": 75.0, "line_total": 150.0},
    ])
    fig_bench = customer_benchmark_bar(rfm_df.iloc[0], rfm_df, currency_symbol="£")
    assert "£150.00" in fig_bench.data[0].text[2]

    fig_timeline = customer_purchase_timeline(tx_df, "C1", currency_symbol="£")
    assert "Order Total (£)" in fig_timeline.layout.yaxis.title.text


def test_compute_rfm_quintiles_deterministic_scoring():
    """Verify compute_rfm_quintiles properly assigns 1-5 scores, reverses recency, and calculates rfm_score and total."""
    df = pd.DataFrame([
        {"customer_id": "C1", "recency": 2, "frequency": 10, "monetary": 1000.0},
        {"customer_id": "C2", "recency": 10, "frequency": 5, "monetary": 500.0},
        {"customer_id": "C3", "recency": 30, "frequency": 3, "monetary": 200.0},
        {"customer_id": "C4", "recency": 60, "frequency": 2, "monetary": 100.0},
        {"customer_id": "C5", "recency": 120, "frequency": 1, "monetary": 20.0},
    ])
    result = compute_rfm_quintiles(df)

    assert "r_score" in result.columns
    assert "f_score" in result.columns
    assert "m_score" in result.columns
    assert "rfm_score" in result.columns
    assert "rfm_total" in result.columns

    r_map = result.set_index("customer_id")
    # C1 is best across all (lowest recency, highest freq, highest spend)
    assert r_map.loc["C1", "r_score"] == 5
    assert r_map.loc["C1", "f_score"] == 5
    assert r_map.loc["C1", "m_score"] == 5
    assert r_map.loc["C1", "rfm_score"] == "555"
    assert r_map.loc["C1", "rfm_total"] == 15

    # C5 is worst across all
    assert r_map.loc["C5", "r_score"] == 1
    assert r_map.loc["C5", "f_score"] == 1
    assert r_map.loc["C5", "m_score"] == 1
    assert r_map.loc["C5", "rfm_score"] == "111"
    assert r_map.loc["C5", "rfm_total"] == 3


def test_compute_rfm_quintiles_handles_ties_without_bin_error():
    """Verify compute_rfm_quintiles handles tie-heavy data (e.g., 90% single-order customers) safely."""
    # 10 customers where 8 have frequency 1
    rows = []
    for i in range(10):
        rows.append({
            "customer_id": f"C{i}",
            "recency": i * 10,
            "frequency": 1 if i < 8 else (i - 6) * 5,
            "monetary": (i + 1) * 25.0,
        })
    df = pd.DataFrame(rows)
    # Should not raise 'ValueError: Bin edges must be unique'
    result = compute_rfm_quintiles(df)
    assert len(result) == 10
    assert result["f_score"].between(1, 5).all()
    assert result["r_score"].between(1, 5).all()


def test_compute_rfm_quintiles_small_dataset_and_empty():
    """Verify compute_rfm_quintiles handles datasets with fewer than 5 rows and empty DataFrames."""
    # Empty
    empty_df = pd.DataFrame(columns=["customer_id", "recency", "frequency", "monetary"])
    res_empty = compute_rfm_quintiles(empty_df)
    assert "rfm_score" in res_empty.columns
    assert len(res_empty) == 0

    # 1 sample
    single_df = pd.DataFrame([{"customer_id": "C1", "recency": 10, "frequency": 1, "monetary": 50.0}])
    res_single = compute_rfm_quintiles(single_df)
    assert res_single.iloc[0]["rfm_score"] == "555"
    assert res_single.iloc[0]["rfm_total"] == 15

    # Missing required column
    invalid_df = pd.DataFrame([{"customer_id": "C1", "recency": 10}])
    with pytest.raises(ValueError, match="Required metric 'frequency' not found"):
        compute_rfm_quintiles(invalid_df)




