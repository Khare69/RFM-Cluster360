"""
Machine Learning module for RFM clustering using K-Means.
Implements log-transformations, standardization, automatic k-selection via silhouette scores, and cluster profiling.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

from config.settings import DEFAULT_K_MAX, DEFAULT_K_MIN, MIN_CUSTOMERS_FOR_CLUSTERING, RANDOM_STATE


def preprocess_rfm(
    rfm_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    scaler_type: str = "standard",
) -> Tuple[np.ndarray, Union[StandardScaler, RobustScaler], List[str]]:
    """
    Applies log1p transformation to squash right-skewed distributions, followed by
    StandardScaler or RobustScaler.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        DataFrame containing RFM metrics.
    features : list of str, optional
        Features to scale. Defaults to ['recency', 'frequency', 'monetary'].
    scaler_type : str, default 'standard'
        Scaler algorithm: 'standard' for StandardScaler, 'robust' for RobustScaler (IQR 25-75).

    Returns
    -------
    Tuple[np.ndarray, Union[StandardScaler, RobustScaler], List[str]]
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

    # Pre-scaling variance guard: check log1p(data).std(axis=0) before scaler forces unit variance
    feature_stds = np.std(log_transformed, axis=0)
    if float(np.max(feature_stds)) < 0.01:
        scaler = StandardScaler(with_std=False)
        scaler.fit(log_transformed)
        scaled_data = np.zeros_like(log_transformed)
    else:
        if scaler_type.lower() == "robust":
            scaler = RobustScaler(quantile_range=(25.0, 75.0))
        else:
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
    # Low-variance guard: check if scaled features have near-zero variance
    if float(scaled_data.std()) < 0.01:
        eval_list = list(eval_range)
        return {
            "k_values": eval_list,
            "silhouette_scores": [0.0] * len(eval_list),
            "inertias": [0.0] * len(eval_list),
            "best_k": eval_list[0],
            "best_silhouette": 0.0,
            "is_low_variance": True,
            "warning": "Dataset has near-zero variance across RFM features. Clustering may not reveal distinct behavioral segments.",
        }

    k_values: List[int] = []
    silhouette_scores: List[float] = []
    inertias: List[float] = []
    calinski_harabasz_scores: List[float] = []
    davies_bouldin_scores: List[float] = []

    for k in eval_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(scaled_data)

        # Silhouette score requires at least 2 distinct clusters
        try:
            if len(np.unique(labels)) > 1:
                sample_sz = 2000 if n_samples > 2000 else None
                sil = float(
                    silhouette_score(
                        scaled_data,
                        labels,
                        sample_size=sample_sz,
                        random_state=random_state,
                    )
                )
                calinski = float(calinski_harabasz_score(scaled_data, labels))
                davies = float(davies_bouldin_score(scaled_data, labels))
            else:
                sil = 0.0
                calinski = 0.0
                davies = 99.0
        except Exception:
            sil = 0.0
            calinski = 0.0
            davies = 99.0

        k_values.append(int(k))
        silhouette_scores.append(round(sil, 4))
        inertias.append(round(float(kmeans.inertia_), 2))
        calinski_harabasz_scores.append(round(calinski, 2))
        davies_bouldin_scores.append(round(davies, 4))

    # Determine best k (highest silhouette, highest calinski, lowest davies)
    best_idx = int(np.argmax(silhouette_scores))
    best_k = k_values[best_idx]
    best_silhouette = silhouette_scores[best_idx]
    best_k_calinski = k_values[int(np.argmax(calinski_harabasz_scores))]
    best_k_davies = k_values[int(np.argmin(davies_bouldin_scores))]

    return {
        "k_values": k_values,
        "silhouette_scores": silhouette_scores,
        "inertias": inertias,
        "calinski_harabasz_scores": calinski_harabasz_scores,
        "davies_bouldin_scores": davies_bouldin_scores,
        "best_k": best_k,
        "best_silhouette": best_silhouette,
        "best_k_calinski": best_k_calinski,
        "best_k_davies": best_k_davies,
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


def align_centroids(
    centroids: np.ndarray,
    anchor_centroids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Prevents cluster label switching across retraining runs and datasets.

    If anchor_centroids are provided and have matching shape, solves minimum-weight
    bipartite matching via Hungarian assignment (linear_sum_assignment) on Euclidean distance.
    Otherwise, applies deterministic composite health rank sorting:
        Health = -Recency_centroid + Frequency_centroid + Monetary_centroid
    ranking 0 (Dormant/Lowest) to k-1 (Champions/Highest).

    Parameters
    ----------
    centroids : np.ndarray
        Array of shape (k, d) representing newly fitted cluster centers.
    anchor_centroids : np.ndarray, optional
        Persisted reference centroids from previous model runs.

    Returns
    -------
    Tuple[np.ndarray, Dict[int, int]]
        (aligned_centroids, label_mapping) where label_mapping maps old_cluster_id -> aligned_cluster_id.
    """
    if anchor_centroids is not None and anchor_centroids.shape == centroids.shape:
        cost_matrix = cdist(centroids, anchor_centroids, metric="euclidean")
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        label_mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    else:
        # Deterministic sort: composite score (-R + F + M)
        if centroids.shape[1] >= 3:
            composite_health = -centroids[:, 0] + centroids[:, 1] + centroids[:, 2]
        else:
            composite_health = np.sum(centroids, axis=1)
        sorted_indices = np.argsort(composite_health)
        label_mapping = {int(old_idx): int(new_idx) for new_idx, old_idx in enumerate(sorted_indices)}

    aligned_centroids = np.zeros_like(centroids)
    for old_i, new_i in label_mapping.items():
        aligned_centroids[new_i] = centroids[old_i]

    return aligned_centroids, label_mapping


def isolate_whales(
    rfm_df: pd.DataFrame,
    quantile: float = 0.995,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes Two-Stage Hybrid Isolation to prevent extreme B2B wholesale buyers
    from distorting K-Means centroids and compressing the 99% retail core.

    Customers with spend or frequency strictly above the specified quantile (e.g., 99.5th percentile)
    are routed to an Institutional / Apex VIP subset.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        DataFrame with 'frequency' and 'monetary' columns.
    quantile : float, default 0.995
        Upper-tail quantile threshold.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (retail_core_df, whales_df)
    """
    if rfm_df.empty or len(rfm_df) < 50:
        return rfm_df.copy(), rfm_df.iloc[0:0].copy()

    f_thresh = float(rfm_df["frequency"].quantile(quantile))
    m_thresh = float(rfm_df["monetary"].quantile(quantile))

    whale_mask = (rfm_df["frequency"] > f_thresh) | (rfm_df["monetary"] > m_thresh)
    whales = rfm_df[whale_mask].copy()
    retail = rfm_df[~whale_mask].copy()

    return retail.reset_index(drop=True), whales.reset_index(drop=True)


def compute_boundary_confidence(
    scaled_data: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Computes cluster assignment confidence score for each customer based on
    the ratio of Euclidean distance to the assigned centroid vs the second-closest centroid:

        Confidence(u) = 1 - ||x_u - c_1|| / ||x_u - c_2||

    Returns an array of floats in [0.0, 1.0], where values near 0.0 indicate boundary ambiguity.

    Parameters
    ----------
    scaled_data : np.ndarray
        Array of shape (n_samples, d).
    centroids : np.ndarray
        Array of shape (k, d).

    Returns
    -------
    np.ndarray
        Array of confidence scores of length n_samples.
    """
    if centroids.shape[0] < 2:
        return np.ones(scaled_data.shape[0], dtype=float)

    dists = cdist(scaled_data, centroids, metric="euclidean")
    sorted_dists = np.sort(dists, axis=1)
    d1 = sorted_dists[:, 0]
    d2 = sorted_dists[:, 1]

    confidence = 1.0 - (d1 / np.maximum(d2, 1e-6))
    return np.clip(confidence, 0.0, 1.0)


def create_rfm_pipeline(
    k: int,
    random_state: int = RANDOM_STATE,
    scaler: Optional[Union[StandardScaler, RobustScaler]] = None,
    kmeans_model: Optional[KMeans] = None,
    scaler_type: str = "standard",
) -> Pipeline:
    """
    Constructs a unified Scikit-Learn Pipeline combining log1p transformation,
    scaling (StandardScaler or RobustScaler), and KMeans clustering.

    Parameters
    ----------
    k : int
        Number of clusters.
    random_state : int, default RANDOM_STATE
        Random state for reproducibility.
    scaler : StandardScaler or RobustScaler, optional
        Pre-configured or pre-fitted scaler. Defaults to a fresh scaler.
    kmeans_model : KMeans, optional
        Pre-configured or pre-fitted KMeans estimator. Defaults to KMeans(n_clusters=k).
    scaler_type : str, default 'standard'
        'standard' for StandardScaler or 'robust' for RobustScaler.

    Returns
    -------
    Pipeline
        Unified Scikit-Learn Pipeline ready for fitting or inference.
    """
    if scaler is not None:
        scaler_step = scaler
    elif scaler_type.lower() == "robust":
        scaler_step = RobustScaler(quantile_range=(25.0, 75.0))
    else:
        scaler_step = StandardScaler()

    kmeans_step = (
        kmeans_model
        if kmeans_model is not None
        else KMeans(n_clusters=k, random_state=random_state, n_init=10)
    )
    return Pipeline([
        ("log1p", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", scaler_step),
        ("kmeans", kmeans_step),
    ])


def fit_rfm_pipeline(
    rfm_df: pd.DataFrame,
    k: int,
    features: Optional[List[str]] = None,
    random_state: int = RANDOM_STATE,
    scaler_type: str = "standard",
) -> Tuple[Pipeline, np.ndarray]:
    """
    Fits a unified Scikit-Learn Pipeline on raw RFM metrics and returns the fitted pipeline and cluster labels.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        DataFrame containing RFM metrics.
    k : int
        Number of clusters to create.
    features : List[str], optional
        Feature column names. Defaults to ['recency', 'frequency', 'monetary'].
    random_state : int, default RANDOM_STATE
        Random state for reproducibility.
    scaler_type : str, default 'standard'
        'standard' or 'robust'.

    Returns
    -------
    Tuple[Pipeline, np.ndarray]
        (fitted_pipeline, cluster_labels)
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

    data = rfm_df[cols].clip(lower=0.0).values
    pipeline = create_rfm_pipeline(k=k, random_state=random_state, scaler_type=scaler_type)
    labels = pipeline.fit_predict(data)
    return pipeline, labels

