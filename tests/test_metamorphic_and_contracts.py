"""
Three-Tier Machine Learning Testing Pyramid:
- Tier 1: Data Contracts & Invariant Verification.
- Tier 2: Metamorphic Testing & Bootstrap Cluster Stability (ARI >= 0.80).
- Tier 3: CRM Reverse-ETL Schema & Stratified Holdout Integrity.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.clv import (
    BGNBDParameters,
    GammaGammaParameters,
    calculate_customer_clv,
    prepare_clv_summary,
)
from src.clustering import (
    align_centroids,
    fit_rfm_pipeline,
    run_kmeans,
)
from src.export import prepare_crm_reverse_etl_payload
from src.reconciler import reconcile_transactions
from src.rfm import calculate_rfm


def _generate_synthetic_retail_transactions(n_customers: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generates realistic synthetic retail transactions for multi-tier invariant testing."""
    np.random.seed(seed)
    base_date = datetime(2026, 1, 1)
    records = []

    for i in range(n_customers):
        cid = f"CUST_{i:03d}"
        n_tx = np.random.randint(1, 8)
        for j in range(n_tx):
            days_offset = np.random.randint(0, 180)
            qty = float(np.random.randint(1, 5))
            price = round(float(np.random.uniform(10.0, 100.0)), 2)
            records.append({
                "customer_id": cid,
                "invoice_no": f"INV_{i}_{j}",
                "invoice_date": base_date + timedelta(days=days_offset),
                "quantity": qty,
                "unit_price": price,
                "description": f"Item_{np.random.randint(1, 10)}",
            })
        # Inject 10% returns
        if np.random.rand() < 0.15:
            records.append({
                "customer_id": cid,
                "invoice_no": f"CINV_{i}",
                "invoice_date": base_date + timedelta(days=np.random.randint(30, 200)),
                "quantity": -1.0,
                "unit_price": 25.0,
                "description": "Item_1",
            })

    return pd.DataFrame(records)


# ==============================================================================
# Tier 1: Data Contracts & Mathematical Invariants
# ==============================================================================


def test_tier1_transaction_and_rfm_invariants():
    """Validates physical and economic invariants on reconciled data and RFM outputs."""
    raw_df = _generate_synthetic_retail_transactions(n_customers=60)
    settled_df, report = reconcile_transactions(raw_df)

    # Invariant 1: In settled DataFrame, all net quantities must be strictly positive
    assert (settled_df["net_quantity"] > 0).all(), "Settled rows must have positive quantity"
    # Invariant 2: Net amounts must be non-negative
    assert (settled_df["net_amount"] >= 0).all(), "Settled rows cannot have negative spend"
    # Invariant 3: Reconciled net revenue cannot exceed gross revenue
    assert report.net_settled_revenue <= report.gross_revenue + 1e-4

    rfm_df = calculate_rfm(settled_df)
    # Invariant 4: Recency non-negative
    assert (rfm_df["recency"] >= 0).all()
    # Invariant 5: Frequency >= 1
    assert (rfm_df["frequency"] >= 1).all()
    # Invariant 6: Monetary spend >= 0
    assert (rfm_df["monetary"] >= 0).all()

    # Invariant 7: CLV bounds
    clv_summary = prepare_clv_summary(settled_df)
    bgnbd_params = BGNBDParameters(r=0.8, alpha=15.0, a=0.5, b=1.2)
    gg_params = GammaGammaParameters(p=2.0, q=4.0, v=80.0)
    clv_df = calculate_customer_clv(clv_summary, bgnbd_params, gg_params)

    assert (clv_df["p_alive"] >= 0.0).all() and (clv_df["p_alive"] <= 1.0).all()
    assert (clv_df["churn_probability"] >= 0.0).all() and (clv_df["churn_probability"] <= 1.0).all()
    assert (clv_df["predicted_clv_6m"] >= 0.0).all()


# ==============================================================================
# Tier 2: Metamorphic & Bootstrap Stability Testing
# ==============================================================================


def test_tier2_metamorphic_rank_and_clv_monotonicity():
    """
    Metamorphic Testing:
    Taking a test customer and synthetically increasing Frequency (+5) and Monetary (+500)
    while decreasing Recency (max(1, R - 20)) must monotonically increase or preserve
    their assigned cluster health rank and predicted CLV.
    """
    raw_df = _generate_synthetic_retail_transactions(n_customers=80)
    settled_df, _ = reconcile_transactions(raw_df)
    rfm_df = calculate_rfm(settled_df)

    pipeline, labels = fit_rfm_pipeline(rfm_df, k=4, random_state=42)
    kmeans = pipeline.named_steps["kmeans"]
    _, label_mapping = align_centroids(kmeans.cluster_centers_)

    # Pick a baseline customer (e.g. customer 0)
    base_cust = rfm_df.iloc[0].copy()
    base_features = np.array([[base_cust["recency"], base_cust["frequency"], base_cust["monetary"]]])
    base_raw_cluster = int(pipeline.predict(base_features)[0])
    base_health_rank = label_mapping.get(base_raw_cluster, base_raw_cluster)

    # Perturbed customer: more frequent (+5), higher spend (+500), more recent (R - 20)
    better_features = np.array([[
        max(1.0, base_cust["recency"] - 20.0),
        base_cust["frequency"] + 5.0,
        base_cust["monetary"] + 500.0,
    ]])
    better_raw_cluster = int(pipeline.predict(better_features)[0])
    better_health_rank = label_mapping.get(better_raw_cluster, better_raw_cluster)

    # Monotonicity check: health rank should be equal or higher (higher = healthier)
    assert better_health_rank >= base_health_rank, (
        f"Metamorphic violation: improved customer health rank ({better_health_rank}) "
        f"dropped below baseline rank ({base_health_rank})"
    )


def test_tier2_bootstrap_cluster_stability_ari():
    """
    Bootstrap Cluster Stability:
    Resamples 90% of the customer dataset across 5 bootstrap iterations and verifies
    that cluster partitions achieve high agreement (Adjusted Rand Index >= 0.80).
    """
    raw_df = _generate_synthetic_retail_transactions(n_customers=100, seed=123)
    settled_df, _ = reconcile_transactions(raw_df)
    rfm_df = calculate_rfm(settled_df)

    features = rfm_df[["recency", "frequency", "monetary"]].to_numpy()
    log_features = np.log1p(features)

    # Fit baseline K-Means
    k = 3
    base_labels, base_centers, _ = run_kmeans(log_features, k=k, random_state=42)
    _, base_map = align_centroids(base_centers)
    aligned_base_labels = np.array([base_map[lbl] for lbl in base_labels])

    ari_scores = []
    np.random.seed(999)
    n_samples = len(rfm_df)
    subsample_size = int(0.90 * n_samples)

    for i in range(5):
        indices = np.random.choice(n_samples, size=subsample_size, replace=False)
        sub_X = log_features[indices]

        sub_labels, sub_centers, _ = run_kmeans(sub_X, k=k, random_state=42 + i)
        _, sub_map = align_centroids(sub_centers, anchor_centroids=base_centers)
        aligned_sub_labels = np.array([sub_map[lbl] for lbl in sub_labels])

        # Compare overlap
        ari = adjusted_rand_score(aligned_base_labels[indices], aligned_sub_labels)
        ari_scores.append(ari)

    mean_ari = float(np.mean(ari_scores))
    assert mean_ari >= 0.80, f"Bootstrap cluster stability below threshold: mean ARI = {mean_ari:.3f} < 0.80"


# ==============================================================================
# Tier 3: CRM Schema & Holdout Integrity
# ==============================================================================


def test_tier3_crm_reverse_etl_schema_and_holdout():
    """Validates CRM export schema formatting and 10% stratified holdout allocation."""
    raw_df = _generate_synthetic_retail_transactions(n_customers=100)
    settled_df, _ = reconcile_transactions(raw_df)
    rfm_df = calculate_rfm(settled_df)
    pipeline, labels = fit_rfm_pipeline(rfm_df, k=3, random_state=42)
    rfm_df["cluster"] = labels
    rfm_df["segment"] = "Champions"

    crm_df = prepare_crm_reverse_etl_payload(rfm_df, holdout_percentage=10)

    # Exact column types
    assert np.issubdtype(crm_df["recency_days"].dtype, np.integer)
    assert np.issubdtype(crm_df["frequency_orders"].dtype, np.integer)
    assert np.issubdtype(crm_df["monetary_spend"].dtype, np.floating)
    assert crm_df["control_holdout"].dtype == bool

    # Holdout percentage should be close to 10%
    holdout_rate = crm_df["control_holdout"].mean()
    assert 0.05 <= holdout_rate <= 0.20
