"""
Unit tests for preprocessing, K-Means clustering, and evaluation metric scoring.
"""

import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    build_clustered_df,
    evaluate_clusters,
    get_cluster_profiles,
    preprocess_rfm,
    run_kmeans,
)


def test_preprocess_rfm(synthetic_rfm_base):
    scaled_data, scaler, cols = preprocess_rfm(synthetic_rfm_base)
    assert scaled_data.shape == (30, 3)
    # StandardScaler guarantees approximately mean 0 and std 1
    assert np.allclose(scaled_data.mean(axis=0), 0, atol=1e-2)
    assert np.allclose(scaled_data.std(axis=0), 1, atol=1e-2)
    assert cols == ["recency", "frequency", "monetary"]


def test_evaluate_clusters(synthetic_rfm_base):
    scaled_data, _, _ = preprocess_rfm(synthetic_rfm_base)
    eval_results = evaluate_clusters(scaled_data, k_range=range(2, 6))

    assert "k_values" in eval_results
    assert "silhouette_scores" in eval_results
    assert "inertias" in eval_results
    assert "best_k" in eval_results

    assert eval_results["k_values"] == [2, 3, 4, 5]
    assert len(eval_results["silhouette_scores"]) == 4
    assert 2 <= eval_results["best_k"] <= 5
    # Inertia should be strictly decreasing with larger k
    inertias = eval_results["inertias"]
    assert all(inertias[i] >= inertias[i+1] for i in range(len(inertias)-1))


def test_run_kmeans_and_build_df(synthetic_rfm_base):
    scaled_data, _, _ = preprocess_rfm(synthetic_rfm_base)
    labels, centers, model = run_kmeans(scaled_data, k=3)

    assert len(labels) == 30
    assert len(np.unique(labels)) == 3
    assert centers.shape == (3, 3)

    clustered_df = build_clustered_df(synthetic_rfm_base, labels)
    assert "cluster" in clustered_df.columns
    assert set(clustered_df["cluster"].unique()) == {0, 1, 2}

    profiles = get_cluster_profiles(clustered_df)
    assert len(profiles) == 3
    assert "customer_share_pct" in profiles.columns
    assert "revenue_share_pct" in profiles.columns
    assert round(profiles["customer_share_pct"].sum(), 0) == 100.0


def test_evaluate_clusters_low_variance():
    """Verify that evaluate_clusters safely returns early with warning when features have near-zero variance."""
    low_var_data = np.zeros((20, 3))
    eval_results = evaluate_clusters(low_var_data)

    assert eval_results.get("is_low_variance") is True
    assert "warning" in eval_results
    assert eval_results["best_silhouette"] == 0.0
    assert eval_results["best_k"] in eval_results["k_values"]
    assert len(eval_results["k_values"]) > 0

