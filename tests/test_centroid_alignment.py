"""
Unit tests for centroid alignment (Hungarian matching), whale isolation,
boundary confidence scoring, and orthogonalized average spend.
"""

from datetime import datetime
import numpy as np
import pandas as pd

from src.clustering import (
    align_centroids,
    compute_boundary_confidence,
    evaluate_clusters,
    isolate_whales,
    preprocess_rfm,
)
from src.rfm import calculate_rfm, check_rfm_collinearity


def test_align_centroids_hungarian_matching():
    # Previous run reference centroids (e.g. k=3)
    anchor_centroids = np.array([
        [-1.0, 1.5, 2.0],  # Cluster 0: Champions
        [0.0, 0.0, 0.0],   # Cluster 1: Potential
        [1.5, -1.0, -1.0], # Cluster 2: Lost
    ])

    # In a new run, K-Means randomly permutes the order:
    # Row 0 is Lost, Row 1 is Champions, Row 2 is Potential
    permuted_centroids = np.array([
        [1.52, -0.98, -1.01],  # Closest to anchor 2
        [-0.99, 1.48, 2.02],   # Closest to anchor 0
        [0.01, -0.02, 0.03],   # Closest to anchor 1
    ])

    aligned_centroids, label_mapping = align_centroids(permuted_centroids, anchor_centroids)

    # Hungarian assignment should map:
    # 0 -> 2 (Lost)
    # 1 -> 0 (Champions)
    # 2 -> 1 (Potential)
    assert label_mapping[0] == 2
    assert label_mapping[1] == 0
    assert label_mapping[2] == 1

    # Aligned centroids row 0 should match anchor row 0
    np.testing.assert_allclose(aligned_centroids[0], permuted_centroids[1])
    np.testing.assert_allclose(aligned_centroids[1], permuted_centroids[2])
    np.testing.assert_allclose(aligned_centroids[2], permuted_centroids[0])


def test_align_centroids_deterministic_sort():
    # Without anchors, composite health = -R + F + M
    centroids = np.array([
        [-1.0, 2.0, 2.0],  # Health: -(-1) + 2 + 2 = 5.0 (Champions -> Highest)
        [1.0, -1.0, -1.0], # Health: -(1) - 1 - 1 = -3.0 (Dormant -> Lowest)
        [0.0, 0.5, 0.5],   # Health: -(0) + 0.5 + 0.5 = 1.0 (Mid)
    ])

    aligned_centroids, label_mapping = align_centroids(centroids, anchor_centroids=None)

    # 1 (health -3.0) -> index 0 (Lowest)
    # 2 (health 1.0) -> index 1
    # 0 (health 5.0) -> index 2 (Highest)
    assert label_mapping[1] == 0
    assert label_mapping[2] == 1
    assert label_mapping[0] == 2


def test_isolate_whales_two_stage():
    # 99 normal customers + 1 extreme whale
    np.random.seed(42)
    n = 100
    freqs = np.random.randint(1, 10, size=n).astype(float)
    spends = np.random.uniform(50, 500, size=n)

    # Inject extreme B2B whale
    freqs[-1] = 500.0
    spends[-1] = 100000.0

    df = pd.DataFrame({
        "customer_id": [f"CUST_{i}" for i in range(n)],
        "frequency": freqs,
        "monetary": spends,
    })

    retail, whales = isolate_whales(df, quantile=0.98)
    assert len(whales) >= 1
    assert "CUST_99" in whales["customer_id"].values
    assert "CUST_99" not in retail["customer_id"].values


def test_compute_boundary_confidence():
    centroids = np.array([
        [0.0, 0.0],
        [10.0, 10.0],
    ])
    # Sample A is exactly at centroid 0 -> dist1 = 0, dist2 ~ 14.14 -> conf ~ 1.0
    # Sample B is midway [5.0, 5.0] -> dist1 == dist2 -> conf ~ 0.0
    scaled_data = np.array([
        [0.0, 0.0],
        [5.0, 5.0],
    ])

    conf = compute_boundary_confidence(scaled_data, centroids)
    assert conf[0] > 0.99
    assert conf[1] < 0.05


def test_multi_index_evaluation():
    np.random.seed(42)
    # 3 distinct clusters in 3D
    c1 = np.random.randn(30, 3) + [0, 0, 0]
    c2 = np.random.randn(30, 3) + [5, 5, 5]
    c3 = np.random.randn(30, 3) + [10, 10, 10]
    X = np.vstack([c1, c2, c3])

    eval_res = evaluate_clusters(X, k_range=range(2, 6))
    assert "calinski_harabasz_scores" in eval_res
    assert "davies_bouldin_scores" in eval_res
    assert len(eval_res["calinski_harabasz_scores"]) == 4
    assert len(eval_res["davies_bouldin_scores"]) == 4
    assert eval_res["best_k_calinski"] in [2, 3, 4, 5]


def test_preprocess_rfm_robust_scaler():
    df = pd.DataFrame({
        "recency": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "frequency": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "monetary": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    })
    scaled, scaler, cols = preprocess_rfm(df, scaler_type="robust")
    assert scaled.shape == (10, 3)
    assert scaler.__class__.__name__ == "RobustScaler"


def test_calculate_rfm_with_average_monetary_and_lookback():
    data = {
        "customer_id": ["C1", "C1", "C2"],
        "invoice_no": ["101", "102", "103"],
        "invoice_date": [datetime(2026, 1, 1), datetime(2026, 1, 15), datetime(2026, 1, 20)],
        "quantity": [2, 3, 1],
        "unit_price": [10.0, 20.0, 50.0],
    }
    df = pd.DataFrame(data)

    # C1: 2 orders, total spend = 2*10 + 3*20 = 80. Frequency = 2. Average spend = 40.
    # C2: 1 order, total spend = 50. Frequency = 1. Average spend = 50.
    rfm_avg = calculate_rfm(df, use_average_monetary=True)
    c1_row = rfm_avg[rfm_avg["customer_id"] == "C1"].iloc[0]
    assert c1_row["frequency"] == 2
    assert c1_row["monetary"] == 40.0

    # Test collinearity check helper
    corr = check_rfm_collinearity(rfm_avg)
    assert "f_m_corr" in corr
    assert "r_f_corr" in corr
