"""
Unit tests for persona labeling and cluster-to-segment classification mapping.
"""

import pandas as pd

from src.clustering import build_clustered_df, preprocess_rfm, run_kmeans
from src.personas import (
    PERSONA_DEFINITIONS,
    SEGMENT_COLORS,
    classify_cluster,
    generate_cluster_label_mapping,
    get_persona_metadata,
    get_segment_color,
    label_segments,
)


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


def test_label_segments_k10_unique_labels():
    """Verify that when k=10 (exceeding 9 base persona labels), all labels remain distinct with ordinal qualifiers."""
    import numpy as np

    np.random.seed(42)
    rows = []
    for cluster_id in range(10):
        for i in range(5):
            rows.append({
                "customer_id": f"C_{cluster_id}_{i}",
                "recency": float(5 + cluster_id * 25),
                "frequency": max(1, 20 - cluster_id * 2),
                "monetary": max(10.0, 5000.0 - cluster_id * 450.0),
                "cluster": cluster_id,
            })
    df = pd.DataFrame(rows)
    cluster_map = generate_cluster_label_mapping(df)

    assert len(cluster_map) == 10
    labels = list(cluster_map.values())
    assert len(set(labels)) == 10, "All 10 cluster labels must be strictly unique"
    assert any("Tier" in label for label in labels), "Overflow cluster should contain a Tier qualifier"


def test_classify_cluster_rule_based_behavior():
    """Verify that classify_cluster maps normalized centroid behavior to correct marketing personas."""
    assert classify_cluster(r_rank=0.9, f_rank=0.8, m_rank=0.85) == "Champions"
    assert classify_cluster(r_rank=0.45, f_rank=0.8, m_rank=0.7) == "Loyal Customers"
    assert classify_cluster(r_rank=0.2, f_rank=0.8, m_rank=0.8) == "At Risk"
    assert classify_cluster(r_rank=0.8, f_rank=0.2, m_rank=0.2) == "New Customers"
    assert classify_cluster(r_rank=0.8, f_rank=0.2, m_rank=0.6) == "Promising"
    assert classify_cluster(r_rank=0.1, f_rank=0.1, m_rank=0.1) == "Lost / Dormant"


def test_get_persona_metadata_and_colors_no_mutation():
    """Verify that resolving tiered persona labels yields complete metadata without mutating global state."""
    initial_keys = set(PERSONA_DEFINITIONS.keys())

    meta_tier2 = get_persona_metadata("Champions (Tier 2)")
    assert "Bought recently" in meta_tier2["description"]
    assert "(Champions (Tier 2))" in meta_tier2["description"]
    assert meta_tier2["color"] == SEGMENT_COLORS["Champions"]

    color_tier3 = get_segment_color("At Risk (Tier 3)")
    assert color_tier3 == SEGMENT_COLORS["At Risk"]

    # Verify bracket notation on SEGMENT_COLORS works seamlessly with Tier qualifiers
    assert SEGMENT_COLORS["Champions (Tier 2)"] == SEGMENT_COLORS["Champions"]
    assert SEGMENT_COLORS["Unknown Segment (Tier 9)"] == "#7f7f7f"

    # Verify that global dictionary keys were not mutated
    assert set(PERSONA_DEFINITIONS.keys()) == initial_keys



