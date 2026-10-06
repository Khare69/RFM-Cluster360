"""
Persona labeling module.
Translates mathematical cluster centroids into actionable, human-readable customer personas and business strategies.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config.settings import SEGMENT_COLORS

# Detailed persona definitions with descriptions and marketing recommendations
PERSONA_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "Champions": {
        "description": "Bought recently, buy often, and spend the most. Your most valuable customers.",
        "action": "Reward them with VIP perks, early product access, and referral incentives.",
        "color": SEGMENT_COLORS.get("Champions", "#1f77b4"),
    },
    "Loyal Customers": {
        "description": "Consistent repeat purchasers with solid lifetime spend and recent engagement.",
        "action": "Upsell higher-value products, offer loyalty program rewards, and ask for reviews.",
        "color": SEGMENT_COLORS.get("Loyal Customers", "#2ca02c"),
    },
    "Potential Loyalists": {
        "description": "Recent customers with multiple purchases and growing spend.",
        "action": "Offer memberships, personalized product recommendations, and engagement discounts.",
        "color": SEGMENT_COLORS.get("Potential Loyalists", "#17becf"),
    },
    "New Customers": {
        "description": "First-time or very recent buyers who have placed 1 or 2 orders.",
        "action": "Send warm onboarding sequences, satisfaction check-ins, and a 2nd purchase incentive.",
        "color": SEGMENT_COLORS.get("New Customers", "#ff7f0e"),
    },
    "Promising": {
        "description": "Recent shoppers who spent an average amount but don't buy frequently yet.",
        "action": "Increase brand awareness with targeted content and seasonal promotions.",
        "color": SEGMENT_COLORS.get("Promising", "#bcbd22"),
    },
    "Needs Attention": {
        "description": "Above-average recency and moderate spend. Showing early signs of disengagement.",
        "action": "Reactivate with limited-time discounts, new arrivals, and re-engagement campaigns.",
        "color": SEGMENT_COLORS.get("Needs Attention", "#9467bd"),
    },
    "About to Sleep": {
        "description": "Below-average recency, frequency, and spend. Will become lost if not re-engaged.",
        "action": "Share popular products, discount vouchers, and reconnect before churn.",
        "color": SEGMENT_COLORS.get("About to Sleep", "#8c564b"),
    },
    "At Risk": {
        "description": "Spent big and bought frequently in the past, but haven't returned in a long time.",
        "action": "Aggressive win-back campaigns, personal outreach, and exclusive return incentives.",
        "color": SEGMENT_COLORS.get("At Risk", "#e377c2"),
    },
    "Lost / Dormant": {
        "description": "Lowest recency, frequency, and monetary values. Inactive for an extended period.",
        "action": "Run low-cost automated win-back campaigns or deprioritize marketing spend.",
        "color": SEGMENT_COLORS.get("Lost / Dormant", "#d62728"),
    },
}


def classify_cluster(
    r_rank: float,  # 0.0 (stale) to 1.0 (recent/fresh)
    f_rank: float,  # 0.0 (low) to 1.0 (high frequency)
    m_rank: float,  # 0.0 (low) to 1.0 (high monetary)
) -> str:
    """
    Classifies a cluster based on normalized percentile ranks of its centroid.

    Parameters
    ----------
    r_rank : float
        Recency score where 1.0 is most recent and 0.0 is least recent.
    f_rank : float
        Frequency score where 1.0 is highest order count and 0.0 is lowest.
    m_rank : float
        Monetary score where 1.0 is highest total spend and 0.0 is lowest.

    Returns
    -------
    str
        Segment name.
    """
    # Top tier: Fresh, frequent, high spenders
    if r_rank >= 0.6 and f_rank >= 0.6 and m_rank >= 0.6:
        return "Champions"

    # High frequency & spend, but freshness is moderate
    if f_rank >= 0.6 and m_rank >= 0.5:
        if r_rank >= 0.4:
            return "Loyal Customers"
        else:
            return "At Risk"

    # High spenders that have been gone a long time
    if m_rank >= 0.7 and r_rank <= 0.4:
        return "At Risk"

    # Fresh buyers with low frequency
    if r_rank >= 0.7 and f_rank <= 0.4:
        if m_rank >= 0.5:
            return "Promising"
        return "New Customers"

    # Moderate fresh and moderate frequent
    if r_rank >= 0.5 and f_rank >= 0.4:
        return "Potential Loyalists"

    # Moderate across the board
    if r_rank >= 0.35 and f_rank >= 0.25:
        return "Needs Attention"

    # Stale, low frequency, low monetary
    if r_rank <= 0.35 and f_rank <= 0.35:
        if m_rank <= 0.35:
            return "Lost / Dormant"
        return "About to Sleep"

    if r_rank <= 0.4:
        return "About to Sleep"

    return "Needs Attention"


ORDERED_PERSONAS: List[str] = [
    "Champions",
    "Loyal Customers",
    "Potential Loyalists",
    "Promising",
    "New Customers",
    "Needs Attention",
    "About to Sleep",
    "At Risk",
    "Lost / Dormant",
]


def generate_cluster_label_mapping(clustered_df: pd.DataFrame) -> Dict[int, str]:
    """
    Generates a deterministic mapping from cluster IDs (0..k-1) to human-readable persona labels.
    Ranks clusters by a weighted composite score (0.35*R_inverted + 0.35*F + 0.30*M) and assigns
    distinct labels in order from the predefined persona hierarchy. Adds ordinal qualifiers
    (e.g., '(Tier 2)') for overflow when k >= 10.

    Parameters
    ----------
    clustered_df : pd.DataFrame
        DataFrame containing 'recency', 'frequency', 'monetary', and 'cluster'.

    Returns
    -------
    Dict[int, str]
        Cluster ID -> Segment Label.
    """
    # Calculate centroids
    centroids = (
        clustered_df.groupby("cluster")
        .agg(
            recency=("recency", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
        )
        .reset_index()
    )

    k = len(centroids)
    if k == 1:
        return {int(centroids["cluster"].iloc[0]): "All Customers"}

    r_vals = centroids["recency"].values
    f_vals = centroids["frequency"].values
    m_vals = centroids["monetary"].values

    def _normalize(arr, invert=False):
        min_v, max_v = arr.min(), arr.max()
        if max_v == min_v:
            return np.full_like(arr, 0.5, dtype=float)
        norm = (arr - min_v) / (max_v - min_v)
        return (1.0 - norm) if invert else norm

    # Recency: lower days is better -> invert ranking so 1.0 is freshest
    r_inverted = _normalize(r_vals, invert=True)
    f_scores = _normalize(f_vals, invert=False)
    m_scores = _normalize(m_vals, invert=False)

    # Weighted composite score: 35% Recency, 35% Frequency, 30% Monetary
    composite_scores = 0.35 * r_inverted + 0.35 * f_scores + 0.30 * m_scores
    centroids["composite_score"] = composite_scores

    # Sort centroids descending by composite score (strongest -> weakest)
    sorted_centroids = centroids.sort_values(by="composite_score", ascending=False).reset_index(drop=True)

    assigned: Dict[int, str] = {}
    used_labels = set()

    for rank, row in sorted_centroids.iterrows():
        c_id = int(row["cluster"])
        persona_idx = int(round(rank * (len(ORDERED_PERSONAS) - 1) / (k - 1)))
        base_label = ORDERED_PERSONAS[persona_idx]

        label = base_label
        if label in used_labels:
            tier = 2
            while f"{base_label} (Tier {tier})" in used_labels:
                tier += 1
            label = f"{base_label} (Tier {tier})"

        assigned[c_id] = label
        used_labels.add(label)

        # Ensure label exists in SEGMENT_COLORS and PERSONA_DEFINITIONS
        if label not in SEGMENT_COLORS:
            SEGMENT_COLORS[label] = SEGMENT_COLORS.get(base_label, "#7f7f7f")
        if label not in PERSONA_DEFINITIONS:
            base_meta = PERSONA_DEFINITIONS.get(base_label, {
                "description": "Customer segment identified through behavioral clustering.",
                "action": "Tailor marketing strategy based on RFM profile.",
                "color": "#7f7f7f",
            })
            PERSONA_DEFINITIONS[label] = {
                "description": f"{base_meta['description']} ({label})",
                "action": base_meta["action"],
                "color": SEGMENT_COLORS[label],
            }

    return assigned


def label_segments(clustered_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Adds a 'segment' column and segment metadata to the clustered DataFrame.

    Parameters
    ----------
    clustered_df : pd.DataFrame
        DataFrame with 'cluster' column.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[int, str]]
        (labeled_df, cluster_to_label_map)
    """
    mapping = generate_cluster_label_mapping(clustered_df)
    labeled_df = clustered_df.copy()
    labeled_df["segment"] = labeled_df["cluster"].map(mapping)
    return labeled_df, mapping
