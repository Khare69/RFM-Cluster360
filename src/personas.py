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


def generate_cluster_label_mapping(clustered_df: pd.DataFrame) -> Dict[int, str]:
    """
    Generates a deterministic mapping from cluster IDs (0..k-1) to human-readable persona labels.
    Ensures all assigned segment names are distinct and meaningful.

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
        return {0: "All Customers"}

    # Compute ranks (0 to 1 scale)
    # Recency: lower days is better -> invert ranking so 1.0 is freshest
    r_vals = centroids["recency"].values
    f_vals = centroids["frequency"].values
    m_vals = centroids["monetary"].values

    def _normalize(arr, invert=False):
        min_v, max_v = arr.min(), arr.max()
        if max_v == min_v:
            return np.full_like(arr, 0.5)
        norm = (arr - min_v) / (max_v - min_v)
        return (1.0 - norm) if invert else norm

    r_scores = _normalize(r_vals, invert=True)  # Low recency = 1.0
    f_scores = _normalize(f_vals, invert=False)
    m_scores = _normalize(m_vals, invert=False)

    assigned: Dict[int, str] = {}
    used_labels = set()

    # Candidate ranking for each cluster
    for i, row in centroids.iterrows():
        c_id = int(row["cluster"])
        r_s, f_s, m_s = r_scores[i], f_scores[i], m_scores[i]
        label = classify_cluster(r_s, f_s, m_s)

        # Disambiguate if label already taken by another cluster
        if label in used_labels:
            # Pick next best semantic fit
            composite_score = 0.4 * r_s + 0.3 * f_s + 0.3 * m_s
            alternatives = [
                ("Champions", composite_score >= 0.75),
                ("Loyal Customers", composite_score >= 0.6 and f_s >= 0.5),
                ("Potential Loyalists", r_s >= 0.6 and composite_score >= 0.5),
                ("New Customers", r_s >= 0.6 and f_s <= 0.3),
                ("Promising", r_s >= 0.5 and m_s >= 0.4),
                ("Needs Attention", composite_score >= 0.35),
                ("At Risk", composite_score < 0.5 and (f_s > 0.4 or m_s > 0.4)),
                ("About to Sleep", composite_score < 0.35 and r_s < 0.4),
                ("Lost / Dormant", composite_score < 0.25),
            ]
            fallback_chosen = False
            for alt_name, cond in alternatives:
                if alt_name not in used_labels and cond:
                    label = alt_name
                    fallback_chosen = True
                    break
            if not fallback_chosen:
                for alt_name, _ in alternatives:
                    if alt_name not in used_labels:
                        label = alt_name
                        break

        assigned[c_id] = label
        used_labels.add(label)

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
