"""
Unit tests for monthly acquisition cohort retention analysis.
"""

import pandas as pd
import pytest

from src.cohort import build_cohort_matrix, get_cohort_summary_stats


def test_build_cohort_matrix():
    # Synthetic transactions over 3 months
    data = [
        # Customer 1: Acquired in Jan (2023-01), bought again in Feb (2023-02) and Mar (2023-03)
        {"customer_id": "C1", "invoice_date": "2023-01-10"},
        {"customer_id": "C1", "invoice_date": "2023-02-15"},
        {"customer_id": "C1", "invoice_date": "2023-03-20"},

        # Customer 2: Acquired in Jan (2023-01), bought again in Feb (2023-02), but not in Mar
        {"customer_id": "C2", "invoice_date": "2023-01-20"},
        {"customer_id": "C2", "invoice_date": "2023-02-10"},

        # Customer 3: Acquired in Feb (2023-02), bought again in Mar (2023-03)
        {"customer_id": "C3", "invoice_date": "2023-02-05"},
        {"customer_id": "C3", "invoice_date": "2023-03-01"},

        # Customer 4: Acquired in Feb (2023-02), never bought again
        {"customer_id": "C4", "invoice_date": "2023-02-18"},
    ]
    df = pd.DataFrame(data)

    retention_matrix, counts_matrix = build_cohort_matrix(df)

    assert "2023-01" in retention_matrix.index
    assert "2023-02" in retention_matrix.index

    # 2023-01 Cohort (C1 and C2 -> 2 customers):
    # Month 0: 100% (2 customers)
    # Month 1 (Feb): 100% (2 customers)
    # Month 2 (Mar): 50% (1 customer)
    jan_ret = retention_matrix.loc["2023-01"]
    assert jan_ret[0] == 100.0
    assert jan_ret[1] == 100.0
    assert jan_ret[2] == 50.0

    # 2023-02 Cohort (C3 and C4 -> 2 customers):
    # Month 0: 100% (2 customers)
    # Month 1 (Mar): 50% (1 customer C3)
    feb_ret = retention_matrix.loc["2023-02"]
    assert feb_ret[0] == 100.0
    assert feb_ret[1] == 50.0

    stats = get_cohort_summary_stats(retention_matrix)
    assert stats["total_cohorts"] == 2
    # Avg M1 retention: (100 + 50) / 2 = 75.0%
    assert stats["avg_m1_retention"] == 75.0


def test_build_cohort_matrix_non_contiguous_months():
    """Verify that cohort matrix fills missing/gap months with NaN rather than skipping columns."""
    data = [
        # Customer 1: Bought in Jan 2023 (Month 0), skipped Feb & Mar, bought again in April 2023 (Month 3)
        {"customer_id": "C1", "invoice_date": "2023-01-10"},
        {"customer_id": "C1", "invoice_date": "2023-04-15"},
    ]
    df = pd.DataFrame(data)

    retention_matrix, counts_matrix = build_cohort_matrix(df)

    assert "2023-01" in retention_matrix.index
    # All intermediate months [0, 1, 2, 3] must exist in columns
    assert list(retention_matrix.columns) == [0, 1, 2, 3]
    jan_ret = retention_matrix.loc["2023-01"]
    assert jan_ret[0] == 100.0
    assert pd.isna(jan_ret[1])
    assert pd.isna(jan_ret[2])
    assert jan_ret[3] == 100.0

