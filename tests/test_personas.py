"""
Unit tests for persona labeling and cluster-to-segment classification mapping.
"""

import pandas as pd
import pytest

from src.clustering import build_clustered_df, preprocess_rfm, run_kmeans
from src.personas import generate_cluster_label_mapping, label_segments


def test_label_segments_with_synthetic(synthetic_rfm_base):
    scaled_data, _, _ = preprocess_rfm(synthetic_rfm_base)
    labels, _, _ = run_kmeans(scaled_data, k=3)
    clustered_df = build_clustered_df(synthetic_rfm_base, labels)

    labeled_df, cluster_map = label_segments(clustered_df)

    assert "segment" in labeled_df.columns
    assert len(cluster_map) == 3
    # Check that each cluster received a non-empty human-readable label
    for c_id, seg_name in cluster_map.items():
        assert isinstance(seg_name, str)
        assert len(seg_name) > 0

    # Ensure champions group (C_CHAMP_*) received a top-tier segment (e.g. Champions or Loyal)
    champ_rows = labeled_df[labeled_df["customer_id"].str.startswith("C_CHAMP_")]
    champ_segments = champ_rows["segment"].unique()
    assert any(s in ["Champions", "Loyal Customers", "Potential Loyalists"] for s in champ_segments)

    # Ensure lost group (C_LOST_*) received an inactive segment (e.g. Lost / Dormant or At Risk or About to Sleep)
    lost_rows = labeled_df[labeled_df["customer_id"].str.startswith("C_LOST_")]
    lost_segments = lost_rows["segment"].unique()
    assert any(s in ["Lost / Dormant", "At Risk", "About to Sleep", "Needs Attention"] for s in lost_segments)
