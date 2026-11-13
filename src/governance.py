"""
First-party data governance, GDPR/CCPA privacy, and machine unlearning module.
Provides HMAC-SHA256 identity pseudonymization and O(1) centroid updating for customer erasure.
"""

import hashlib
import hmac
import secrets
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


def generate_secret_salt() -> str:
    """Generates a secure cryptographically random 32-byte hex salt."""
    return secrets.token_hex(32)


def pseudonymize_customer_ids(
    df: pd.DataFrame,
    salt: Optional[str] = None,
    customer_id_col: str = "customer_id",
) -> Tuple[pd.DataFrame, str]:
    """
    Pseudonymizes customer identifiers using HMAC-SHA256 with an injected salt.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing customer identifiers.
    salt : str, optional
        Secret salt for HMAC hashing. If None, a new random salt is generated.
    customer_id_col : str, default 'customer_id'
        Column name to pseudonymize.

    Returns
    -------
    Tuple[pd.DataFrame, str]
        (DataFrame with hashed customer_ids, secret_salt_used)
    """
    if customer_id_col not in df.columns:
        raise ValueError(f"Column '{customer_id_col}' not found in DataFrame.")

    used_salt = salt if salt is not None else generate_secret_salt()
    salt_bytes = used_salt.encode("utf-8")

    def _hash_id(cid):
        if pd.isna(cid):
            return cid
        cid_bytes = str(cid).strip().encode("utf-8")
        h = hmac.new(salt_bytes, cid_bytes, hashlib.sha256)
        # 16-char hex prefix is sufficient for unique identity resolution while preserving readability
        return f"usr_{h.hexdigest()[:16]}"

    result_df = df.copy()
    result_df[customer_id_col] = result_df[customer_id_col].apply(_hash_id)
    return result_df, used_salt


def forget_user(
    customer_id: str,
    clustered_df: pd.DataFrame,
    centroids: np.ndarray,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    Executes GDPR Article 17 (Right to Erasure) machine unlearning by purging the user
    from the customer table and updating the K-Means cluster centroid in O(1) time:

        c_k_new = (N_k * c_k_old - x_u) / (N_k - 1)

    Parameters
    ----------
    customer_id : str
        Identifier of the customer requesting erasure.
    clustered_df : pd.DataFrame
        Clustered customer DataFrame containing 'customer_id', 'cluster', and features.
    centroids : np.ndarray
        Array of shape (k, d) holding current cluster centroids.
    feature_cols : List[str], optional
        Feature names corresponding to the centroid dimensions. Defaults to ['recency', 'frequency', 'monetary'].

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, int]
        (updated_clustered_df, updated_centroids, affected_cluster_id)
    """
    cols = feature_cols if feature_cols is not None else ["recency", "frequency", "monetary"]
    cid_str = str(customer_id).strip()
    match_mask = clustered_df["customer_id"].astype(str).str.strip() == cid_str

    if not match_mask.any():
        raise KeyError(f"Customer '{customer_id}' not found in clustered dataset.")

    user_row = clustered_df[match_mask].iloc[0]
    cluster_id = int(user_row["cluster"])
    user_vector = user_row[cols].to_numpy(dtype=float)

    # Compute current cluster count N_k
    k_mask = clustered_df["cluster"] == cluster_id
    n_k = int(k_mask.sum())

    updated_centroids = centroids.copy()

    if n_k > 1:
        # Exact O(1) arithmetic mean update
        old_centroid = centroids[cluster_id]
        new_centroid = (n_k * old_centroid - user_vector) / (n_k - 1)
        updated_centroids[cluster_id] = new_centroid
    else:
        # If the cluster only had this user, keep centroid or set to zeros
        updated_centroids[cluster_id] = np.zeros_like(centroids[cluster_id])

    updated_df = clustered_df[~match_mask].reset_index(drop=True)
    return updated_df, updated_centroids, cluster_id
