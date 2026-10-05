"""
Machine Learning module for RFM clustering using K-Means.
Implements log-transformations, standardization, automatic k-selection via silhouette scores, and cluster profiling.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config.settings import DEFAULT_K_MAX, DEFAULT_K_MIN, MIN_CUSTOMERS_FOR_CLUSTERING, RANDOM_STATE


def preprocess_rfm(
    rfm_df: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """
    Applies log1p transformation to squash right-skewed distributions, followed by StandardScaler.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        DataFrame containing RFM metrics.
    features : list of str, optional
        Features to scale. Defaults to ['recency', 'frequency', 'monetary'].

    Returns
    -------
    Tuple[np.ndarray, StandardScaler, List[str]]
        (scaled_features_array, fitted_scaler, feature_names)
    """
    cols = features if features is not None else ["recency", "frequency", "monetary"]
    for col in cols:
        if col not in rfm_df.columns:
            raise ValueError(f"Feature '{col}' not found in RFM DataFrame.")

    if len(rfm_df) < MIN_CUSTOMERS_FOR_CLUSTERING:
        raise ValueError(
            f"Dataset has only {len(rfm_df)} customers. "
            f"A minimum of {MIN_CUSTOMERS_FOR_CLUSTERING} customers is required for meaningful clustering."
        )

    # Extract numerical matrix
    data = rfm_df[cols].copy()

    # Clip negative values if any (monetary or recency cannot be negative for log)
    for col in cols:
        data[col] = data[col].clip(lower=0)

    # Log1p transformation to handle skewness
    log_transformed = np.log1p(data.values)

    # StandardScaler to bring all 3 dimensions to standard normal (mean=0, variance=1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(log_transformed)

    return scaled_data, scaler, cols


def evaluate_clusters(
    scaled_data: np.ndarray,
    k_range: Optional[range] = None,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Evaluates K-Means clustering across a range of k values using Inertia (Elbow method) and Silhouette scores.

    Parameters
    ----------
    scaled_data : np.ndarray
        Standardized RFM feature array.
    k_range : range, optional
        Range of k values to evaluate. Defaults to range(DEFAULT_K_MIN, DEFAULT_K_MAX + 1).
    random_state : int, default RANDOM_STATE
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - 'k_values': list of ints
        - 'silhouette_scores': list of floats
        - 'inertias': list of floats
        - 'best_k': int (k with highest silhouette score)
        - 'best_silhouette': float
    """
    n_samples = scaled_data.shape[0]
    if k_range is None:
        max_k = min(DEFAULT_K_MAX, n_samples - 1)
        min_k = min(DEFAULT_K_MIN, max_k)
        eval_range = range(min_k, max_k + 1)
    else:
        # Cap k_range by n_samples
        valid_ks = [k for k in k_range if 2 <= k < n_samples]
        if not valid_ks:
            raise ValueError(f"Insufficient samples ({n_samples}) for evaluated k_range.")
        eval_range = valid_ks

    k_values: List[int] = []
    silhouette_scores: List[float] = []
    inertias: List[float] = []

    for k in eval_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(scaled_data)

        # Silhouette score requires at least 2 distinct clusters
        try:
            if len(np.unique(labels)) > 1:
                sil = float(silhouette_score(scaled_data, labels))
            else:
                sil = 0.0
        except Exception:
            sil = 0.0

        k_values.append(int(k))
        silhouette_scores.append(round(sil, 4))
        inertias.append(round(float(kmeans.inertia_), 2))

    # Determine best k (highest silhouette)
    best_idx = int(np.argmax(silhouette_scores))
    best_k = k_values[best_idx]
    best_silhouette = silhouette_scores[best_idx]

    return {
        "k_values": k_values,
        "silhouette_scores": silhouette_scores,
        "inertias": inertias,
        "best_k": best_k,
        "best_silhouette": best_silhouette,
    }


def run_kmeans(
    scaled_data: np.ndarray,
    k: int,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, KMeans]:
    """
    Fits K-Means algorithm with specified number of clusters.

    Parameters
    ----------
    scaled_data : np.ndarray
        Standardized feature matrix.
    k : int
        Number of clusters to create.
    random_state : int, default RANDOM_STATE
        Random seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, KMeans]
        (cluster_labels, cluster_centers, fitted_kmeans_model)
    """
    if k < 2 or k >= scaled_data.shape[0]:
        raise ValueError(f"Invalid cluster count k={k} for {scaled_data.shape[0]} samples.")

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    centers = kmeans.cluster_centers_

    return labels, centers, kmeans


def build_clustered_df(
    rfm_df: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Appends cluster assignments to the RFM DataFrame.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        RFM DataFrame.
    labels : np.ndarray
        Cluster indices (0 to k-1).

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'cluster' column.
    """
    clustered_df = rfm_df.copy()
    clustered_df["cluster"] = labels
    return clustered_df


def get_cluster_profiles(clustered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates summary statistics per cluster: customer count, revenue share, mean and median RFM.

    Parameters
    ----------
    clustered_df : pd.DataFrame
        DataFrame with RFM metrics and 'cluster' column.

    Returns
    -------
    pd.DataFrame
        Summary table indexed by cluster.
    """
    total_customers = len(clustered_df)
    total_revenue = clustered_df["monetary"].sum()

    profiles = clustered_df.groupby("cluster").agg(
        customer_count=("customer_id", "count"),
        total_revenue=("monetary", "sum"),
        mean_recency=("recency", "mean"),
        median_recency=("recency", "median"),
        mean_frequency=("frequency", "mean"),
        median_frequency=("frequency", "median"),
        mean_monetary=("monetary", "mean"),
        median_monetary=("monetary", "median"),
    ).reset_index()

    profiles["customer_share_pct"] = np.round((profiles["customer_count"] / total_customers) * 100, 1)
    profiles["revenue_share_pct"] = np.round((profiles["total_revenue"] / total_revenue) * 100, 1) if total_revenue > 0 else 0.0

    # Format floats
    for col in ["mean_recency", "median_recency", "mean_frequency", "median_frequency", "mean_monetary", "median_monetary", "total_revenue"]:
        profiles[col] = profiles[col].round(2)

    return profiles
