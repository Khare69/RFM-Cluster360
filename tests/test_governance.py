"""
Unit tests for GDPR/CCPA pseudonymization and machine unlearning (O(1) erasure).
"""

import numpy as np
import pandas as pd
import pytest

from src.governance import forget_user, pseudonymize_customer_ids


def test_pseudonymize_customer_ids_deterministic():
    df = pd.DataFrame({
        "customer_id": ["12345", "67890", "12345"],
        "spend": [100.0, 200.0, 50.0],
    })
    salt = "my_fixed_testing_salt_1234567890abcdef"
    hashed_df, used_salt = pseudonymize_customer_ids(df, salt=salt)

    assert used_salt == salt
    assert hashed_df["customer_id"].iloc[0].startswith("usr_")
    # Same customer_id produces identical hash
    assert hashed_df["customer_id"].iloc[0] == hashed_df["customer_id"].iloc[2]
    # Distinct customer_ids produce different hashes
    assert hashed_df["customer_id"].iloc[0] != hashed_df["customer_id"].iloc[1]


def test_forget_user_o1_centroid_update():
    # 3 customers in cluster 0
    # Customer 1: [10, 2, 100]
    # Customer 2: [20, 4, 200]
    # Customer 3: [30, 6, 300]
    # Mean of cluster 0: [20, 4, 200]
    df = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "cluster": [0, 0, 0],
        "recency": [10.0, 20.0, 30.0],
        "frequency": [2.0, 4.0, 6.0],
        "monetary": [100.0, 200.0, 300.0],
    })
    initial_centroids = np.array([[20.0, 4.0, 200.0]])

    # User C3 requests GDPR erasure:
    # Remaining: C1 [10, 2, 100] and C2 [20, 4, 200]
    # New expected mean: [(10+20)/2, (2+4)/2, (100+200)/2] = [15.0, 3.0, 150.0]
    updated_df, updated_centroids, affected_cluster = forget_user(
        customer_id="C3",
        clustered_df=df,
        centroids=initial_centroids,
        feature_cols=["recency", "frequency", "monetary"],
    )

    assert len(updated_df) == 2
    assert "C3" not in updated_df["customer_id"].values
    assert affected_cluster == 0
    np.testing.assert_allclose(updated_centroids[0], [15.0, 3.0, 150.0])


def test_forget_user_not_found():
    df = pd.DataFrame({
        "customer_id": ["C1"],
        "cluster": [0],
        "recency": [10.0],
        "frequency": [2.0],
        "monetary": [100.0],
    })
    centroids = np.array([[10.0, 2.0, 100.0]])
    with pytest.raises(KeyError):
        forget_user("NON_EXISTENT", df, centroids)
