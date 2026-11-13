"""
Unit tests for probabilistic Customer Lifetime Value (BG/NBD + Gamma-Gamma)
and Reverse-ETL CRM export formatting.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.clv import (
    BGNBDParameters,
    GammaGammaParameters,
    calculate_customer_clv,
    calculate_p_alive,
    fit_bgnbd,
    fit_gamma_gamma,
    predict_conditional_spend,
    predict_future_transactions,
    prepare_clv_summary,
)
from src.export import prepare_crm_reverse_etl_payload


def test_prepare_clv_summary():
    base_date = datetime(2026, 1, 1)
    data = {
        "customer_id": ["C1", "C1", "C2"],
        "invoice_no": ["INV1", "INV2", "INV3"],
        "invoice_date": [base_date, base_date + timedelta(days=30), base_date + timedelta(days=10)],
        "quantity": [1, 2, 1],
        "unit_price": [50.0, 50.0, 100.0],
    }
    df = pd.DataFrame(data)
    summary = prepare_clv_summary(df)

    assert len(summary) == 2
    c1 = summary[summary["customer_id"] == "C1"].iloc[0]
    # C1 has 2 orders -> frequency_repeat = 1
    assert c1["frequency_repeat"] == 1
    assert c1["recency_days"] == 30.0

    c2 = summary[summary["customer_id"] == "C2"].iloc[0]
    # C2 has 1 order -> frequency_repeat = 0
    assert c2["frequency_repeat"] == 0
    assert c2["recency_days"] == 0.0


def test_bgnbd_fit_and_p_alive_monotonicity():
    # Customer A is active (repeat purchases, very recent)
    # Customer B is dormant (bought long ago, no recent activity)
    summary_df = pd.DataFrame({
        "customer_id": ["Active", "Dormant"],
        "frequency_repeat": [5, 5],
        "recency_days": [150.0, 10.0],  # Active latest order day 150 of 160; Dormant latest day 10 of 160
        "tenure_days": [160.0, 160.0],
        "monetary_avg": [100.0, 100.0],
    })

    params = fit_bgnbd(summary_df)
    assert isinstance(params, BGNBDParameters)
    assert params.r > 0
    assert params.alpha > 0

    p_alive = calculate_p_alive(summary_df, params)
    assert p_alive[0] > p_alive[1], "Active customer should have strictly higher P(Alive) than dormant customer"

    future_tx = predict_future_transactions(summary_df, params, t_days=180)
    assert future_tx[0] > future_tx[1]


def test_gamma_gamma_empirical_bayes_shrinkage():
    summary_df = pd.DataFrame({
        "customer_id": ["Repeat1", "Repeat2", "OneTimeLow", "OneTimeHigh"],
        "frequency_repeat": [3, 4, 0, 0],
        "recency_days": [50.0, 60.0, 0.0, 0.0],
        "tenure_days": [100.0, 100.0, 100.0, 100.0],
        "monetary_avg": [200.0, 220.0, 10.0, 1000.0],
    })

    gg_params = fit_gamma_gamma(summary_df)
    assert isinstance(gg_params, GammaGammaParameters)

    expected_spends = predict_conditional_spend(summary_df, gg_params, kappa=1.0)
    assert len(expected_spends) == 4

    # OneTimeHigh spend ($1000) should be shrunk downward toward population median
    assert expected_spends[3] < 1000.0
    # OneTimeLow spend ($10) should be shrunk upward toward population median
    assert expected_spends[2] > 10.0


def test_calculate_customer_clv_end_to_end():
    summary_df = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "frequency_repeat": [2, 0],
        "recency_days": [30.0, 0.0],
        "tenure_days": [60.0, 60.0],
        "monetary_avg": [150.0, 50.0],
    })
    bgnbd_params = BGNBDParameters(r=0.8, alpha=15.0, a=0.5, b=1.2)
    gg_params = GammaGammaParameters(p=2.0, q=4.0, v=100.0)

    clv_df = calculate_customer_clv(summary_df, bgnbd_params, gg_params, horizon_days=180.0)
    assert "predicted_clv_6m" in clv_df.columns
    assert "p_alive" in clv_df.columns
    assert "churn_probability" in clv_df.columns
    assert clv_df["predicted_clv_6m"].iloc[0] > 0.0


def test_prepare_crm_reverse_etl_payload():
    labeled_df = pd.DataFrame({
        "customer_id": [f"CUST_{i}" for i in range(100)],
        "recency": [15.2] * 100,
        "frequency": [4.0] * 100,
        "monetary": [250.75] * 100,
        "segment": ["Champions"] * 100,
        "cluster": [1] * 100,
        "predicted_clv_6m": [450.00] * 100,
        "churn_probability": [0.15] * 100,
    })

    crm_payload = prepare_crm_reverse_etl_payload(labeled_df, holdout_percentage=10)
    assert len(crm_payload) == 100
    assert "control_holdout" in crm_payload.columns
    # Approximately 10% should be holdout
    holdout_count = int(crm_payload["control_holdout"].sum())
    assert 5 <= holdout_count <= 20
    assert crm_payload["recency_days"].dtype == int or np.issubdtype(crm_payload["recency_days"].dtype, np.integer)
    assert crm_payload["frequency_orders"].dtype == int or np.issubdtype(crm_payload["frequency_orders"].dtype, np.integer)
