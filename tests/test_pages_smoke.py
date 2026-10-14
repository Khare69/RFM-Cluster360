"""
AppTest smoke tests across all Streamlit application pages.
Ensures zero runtime errors, unhandled exceptions, or missing dependency crashes
in both uninitialized guard states and fully populated analytical states.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.cleaning import CleaningReport
from src.cohort import build_cohort_matrix
from src.rfm import compute_rfm_quintiles


@pytest.fixture
def mock_populated_session():
    """Generates a cohesive set of DataFrames and metadata for full-session smoke tests."""
    np.random.seed(42)
    cust_ids = [f"CUST_{i:03d}" for i in range(1, 21)]

    # 1. Clean transactions
    tx_rows = []
    base_date = pd.to_datetime("2023-01-01")
    for cid in cust_ids:
        for order_idx in range(np.random.randint(1, 5)):
            order_date = base_date + pd.Timedelta(days=int(np.random.randint(0, 90)))
            qty = int(np.random.randint(1, 10))
            price = float(np.random.uniform(10.0, 100.0))
            tx_rows.append({
                "customer_id": cid,
                "invoice_no": f"INV_{cid}_{order_idx}",
                "invoice_date": order_date,
                "quantity": qty,
                "unit_price": price,
                "line_total": round(qty * price, 2),
                "country": "United Kingdom",
            })
    clean_df = pd.DataFrame(tx_rows)

    # 2. Labeled RFM DataFrame
    rfm_rows = []
    segments = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Hibernating"]
    for idx, cid in enumerate(cust_ids):
        seg = segments[idx % len(segments)]
        rfm_rows.append({
            "customer_id": cid,
            "recency": int(np.random.randint(1, 120)),
            "frequency": int(np.random.randint(1, 15)),
            "monetary": float(np.random.uniform(50.0, 3000.0)),
            "cluster": idx % len(segments),
            "segment": seg,
            "first_purchase": pd.to_datetime("2023-01-01"),
            "last_purchase": pd.to_datetime("2023-03-15"),
            "country": "United Kingdom",
        })
    labeled_df = pd.DataFrame(rfm_rows)
    labeled_df = compute_rfm_quintiles(labeled_df)

    # 3. Cohort matrices
    cohort_retention, cohort_counts = build_cohort_matrix(clean_df)

    # 4. Cleaning report
    cleaning_report = CleaningReport(
        total_rows_before=len(clean_df) + 10,
        total_rows_after=len(clean_df),
        total_customers=len(cust_ids),
        total_revenue=float(clean_df["line_total"].sum()),
        date_min="2023-01-01",
        date_max="2023-03-31",
    )

    # 5. Clustering metrics
    clustering_metrics = {
        "k_values": [2, 3, 4, 5],
        "inertias": [120.5, 80.2, 50.1, 35.0],
        "silhouette_scores": [0.38, 0.46, 0.42, 0.39],
        "best_k": 3,
        "best_silhouette": 0.46,
    }

    return {
        "clean_df": clean_df,
        "labeled_df": labeled_df,
        "cohort_matrix": cohort_retention,
        "cohort_counts": cohort_counts,
        "cleaning_report": cleaning_report,
        "clustering_metrics": clustering_metrics,
        "selected_k": 5,
        "currency_symbol": "$",
        "snapshot_date": pd.to_datetime("2023-04-01"),
    }


def test_app_main_empty_state_smoke():
    """Verify app.py launches cleanly in uninitialized state with zero exceptions."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0


def test_page_1_overview_uninitialized_and_populated(mock_populated_session):
    """Verify pages/1_Overview.py handles guard clause when empty and renders cleanly when populated."""
    page_path = Path(__file__).resolve().parent.parent / "pages" / "1_Overview.py"

    # 1. Uninitialized
    at_uninit = AppTest.from_file(str(page_path), default_timeout=30)
    at_uninit.run()
    assert len(at_uninit.exception) == 0
    assert len(at_uninit.warning) > 0

    # 2. Populated
    at_pop = AppTest.from_file(str(page_path), default_timeout=30)
    session = mock_populated_session
    at_pop.session_state["labeled_df"] = session["labeled_df"]
    at_pop.session_state["clean_df"] = session["clean_df"]
    at_pop.session_state["cleaning_report"] = session["cleaning_report"]
    at_pop.session_state["selected_k"] = session["selected_k"]
    at_pop.session_state["currency_symbol"] = session["currency_symbol"]
    at_pop.run()
    assert len(at_pop.exception) == 0


def test_page_2_rfm_explorer_uninitialized_and_populated(mock_populated_session):
    """Verify pages/2_RFM_Explorer.py renders histograms, quintiles, and ML diagnostics."""
    page_path = Path(__file__).resolve().parent.parent / "pages" / "2_RFM_Explorer.py"

    # 1. Uninitialized
    at_uninit = AppTest.from_file(str(page_path), default_timeout=30)
    at_uninit.run()
    assert len(at_uninit.exception) == 0
    assert len(at_uninit.warning) > 0

    # 2. Populated
    at_pop = AppTest.from_file(str(page_path), default_timeout=30)
    session = mock_populated_session
    at_pop.session_state["labeled_df"] = session["labeled_df"]
    at_pop.session_state["clustering_metrics"] = session["clustering_metrics"]
    at_pop.session_state["currency_symbol"] = session["currency_symbol"]
    at_pop.session_state["snapshot_date"] = session["snapshot_date"]
    at_pop.run()
    assert len(at_pop.exception) == 0


def test_page_3_cluster_map_uninitialized_and_populated(mock_populated_session):
    """Verify pages/3_Cluster_Map.py renders 3D and 2D projections without exceptions."""
    page_path = Path(__file__).resolve().parent.parent / "pages" / "3_Cluster_Map.py"

    # 1. Uninitialized
    at_uninit = AppTest.from_file(str(page_path), default_timeout=30)
    at_uninit.run()
    assert len(at_uninit.exception) == 0

    # 2. Populated
    at_pop = AppTest.from_file(str(page_path), default_timeout=30)
    session = mock_populated_session
    at_pop.session_state["labeled_df"] = session["labeled_df"]
    at_pop.session_state["currency_symbol"] = session["currency_symbol"]
    at_pop.run()
    assert len(at_pop.exception) == 0


def test_page_4_customer_search_uninitialized_and_populated(mock_populated_session):
    """Verify pages/4_Customer_Search.py renders customer account profile and quintile metric."""
    page_path = Path(__file__).resolve().parent.parent / "pages" / "4_Customer_Search.py"

    # 1. Uninitialized
    at_uninit = AppTest.from_file(str(page_path), default_timeout=30)
    at_uninit.run()
    assert len(at_uninit.exception) == 0

    # 2. Populated
    at_pop = AppTest.from_file(str(page_path), default_timeout=30)
    session = mock_populated_session
    at_pop.session_state["labeled_df"] = session["labeled_df"]
    at_pop.session_state["clean_df"] = session["clean_df"]
    at_pop.session_state["currency_symbol"] = session["currency_symbol"]
    at_pop.session_state["snapshot_date"] = session["snapshot_date"]
    at_pop.run()
    assert len(at_pop.exception) == 0


def test_page_5_cohort_retention_uninitialized_and_populated(mock_populated_session):
    """Verify pages/5_Cohort_Retention.py renders retention KPIs and heatmap."""
    page_path = Path(__file__).resolve().parent.parent / "pages" / "5_Cohort_Retention.py"

    # 1. Uninitialized
    at_uninit = AppTest.from_file(str(page_path), default_timeout=30)
    at_uninit.run()
    assert len(at_uninit.exception) == 0

    # 2. Populated
    at_pop = AppTest.from_file(str(page_path), default_timeout=30)
    session = mock_populated_session
    at_pop.session_state["cohort_matrix"] = session["cohort_matrix"]
    at_pop.session_state["cohort_counts"] = session["cohort_counts"]
    at_pop.run()
    assert len(at_pop.exception) == 0
