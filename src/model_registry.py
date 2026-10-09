"""
MLOps Model Artifact Registry module.
Provides serialization, versioning, persistence, inspection, and inference capabilities
for trained K-Means clustering models and StandardScaler transformations.
"""

from datetime import datetime, timezone
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config.settings import MODEL_REGISTRY_DIR

# Standard feature set
DEFAULT_FEATURES: List[str] = ["recency", "frequency", "monetary"]


def get_default_registry_dir() -> str:
    """Returns absolute path to the default model registry directory."""
    try:
        from config.settings import MODEL_REGISTRY_DIR
        return MODEL_REGISTRY_DIR
    except (ImportError, AttributeError):
        return os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "models")
        )


def _get_next_version_id(save_dir: str, k: int) -> str:
    """
    Computes a sequential, human-readable version identifier like 'v1_k4_20261009_163000'.
    Scans existing directories to determine the next version index.
    """
    os.makedirs(save_dir, exist_ok=True)
    existing_dirs = [
        d for d in os.listdir(save_dir)
        if os.path.isdir(os.path.join(save_dir, d))
    ]

    highest_idx = 0
    for d in existing_dirs:
        match = re.match(r"^v(\d+)_", d)
        if match:
            idx = int(match.group(1))
            if idx > highest_idx:
                highest_idx = idx

    next_idx = highest_idx + 1
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"v{next_idx}_k{k}_{timestamp}"


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively converts NumPy scalars and non-standard types into JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, range)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    elif obj is None or isinstance(obj, (str, bool)):
        return obj
    return str(obj)


def save_model_artifact(
    model: KMeans,
    scaler: StandardScaler,
    cluster_map: Dict[int, str],
    eval_metrics: Optional[Dict[str, Any]] = None,
    feature_names: Optional[List[str]] = None,
    customer_count: Optional[int] = None,
    notes: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> str:
    """
    Serializes and persists a trained KMeans model, StandardScaler, and metadata as a versioned artifact.

    Parameters
    ----------
    model : KMeans
        Fitted Scikit-Learn KMeans clustering model.
    scaler : StandardScaler
        Fitted StandardScaler instance used to preprocess RFM features.
    cluster_map : Dict[int, str]
        Mapping from cluster ID (0..k-1) to persona label.
    eval_metrics : Dict[str, Any], optional
        Evaluation metrics (silhouette score, inertia, elbow curve data).
    feature_names : List[str], optional
        List of feature column names used for training (default: ['recency', 'frequency', 'monetary']).
    customer_count : int, optional
        Total customer records used to train this model.
    notes : str, optional
        User-provided description or tag.
    save_dir : str, optional
        Target directory for model registry. Defaults to config `MODEL_REGISTRY_DIR`.

    Returns
    -------
    str
        Path to the created version directory.
    """
    if not isinstance(model, KMeans):
        raise TypeError("model must be an instance of sklearn.cluster.KMeans")
    if not isinstance(scaler, StandardScaler):
        raise TypeError("scaler must be an instance of sklearn.preprocessing.StandardScaler")
    if not isinstance(cluster_map, dict):
        raise TypeError("cluster_map must be a dictionary")

    features = feature_names if feature_names is not None else DEFAULT_FEATURES
    registry_dir = save_dir if save_dir is not None else get_default_registry_dir()

    k = int(model.n_clusters)
    version_id = _get_next_version_id(registry_dir, k)
    version_dir = os.path.join(registry_dir, version_id)
    os.makedirs(version_dir, exist_ok=True)

    # 1. Serialize KMeans model
    model_path = os.path.join(version_dir, "kmeans_model.joblib")
    joblib.dump(model, model_path)

    # 2. Serialize StandardScaler
    scaler_path = os.path.join(version_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # 3. Extract metrics
    inertia = float(getattr(model, "inertia_", 0.0))
    silhouette: Optional[float] = None
    if eval_metrics:
        if "best_silhouette" in eval_metrics and eval_metrics.get("best_k") == k:
            silhouette = float(eval_metrics["best_silhouette"])
        elif "silhouette_scores" in eval_metrics and "k_values" in eval_metrics:
            try:
                idx = list(eval_metrics["k_values"]).index(k)
                silhouette = float(eval_metrics["silhouette_scores"][idx])
            except (ValueError, IndexError):
                silhouette = float(eval_metrics.get("best_silhouette", 0.0))

    # 4. Prepare Metadata
    metadata: Dict[str, Any] = {
        "version_id": version_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "inertia": round(inertia, 4),
        "silhouette_score": round(silhouette, 4) if silhouette is not None else None,
        "feature_names": list(features),
        "customer_count": int(customer_count) if customer_count is not None else None,
        "cluster_map": {str(c_id): label for c_id, label in cluster_map.items()},
        "eval_metrics": _sanitize_for_json(eval_metrics) if eval_metrics else None,
        "notes": notes or "Auto-saved pipeline artifact",
    }

    metadata_path = os.path.join(version_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return version_dir


def list_model_versions(save_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Lists all model versions saved in the registry with their metadata.

    Parameters
    ----------
    save_dir : str, optional
        Target directory for model registry. Defaults to config `MODEL_REGISTRY_DIR`.

    Returns
    -------
    List[Dict[str, Any]]
        List of metadata dictionaries, sorted newest first.
    """
    registry_dir = save_dir if save_dir is not None else get_default_registry_dir()
    if not os.path.exists(registry_dir):
        return []

    versions: List[Dict[str, Any]] = []
    for item in os.listdir(registry_dir):
        v_path = os.path.join(registry_dir, item)
        if not os.path.isdir(v_path):
            continue

        meta_file = os.path.join(v_path, "metadata.json")
        model_file = os.path.join(v_path, "kmeans_model.joblib")
        scaler_file = os.path.join(v_path, "scaler.joblib")

        if os.path.exists(meta_file) and os.path.exists(model_file) and os.path.exists(scaler_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["path"] = v_path
                meta["version_id"] = meta.get("version_id", item)
                # Convert cluster_map keys to integers for easy consumption
                if "cluster_map" in meta and isinstance(meta["cluster_map"], dict):
                    meta["cluster_map_int"] = {int(k): v for k, v in meta["cluster_map"].items()}
                versions.append(meta)
            except Exception:
                # Skip corrupted metadata files
                continue

    # Sort descending by creation timestamp or folder name
    versions.sort(key=lambda x: x.get("created_at", x.get("version_id", "")), reverse=True)
    return versions


def load_model_artifact(
    version_dir: str,
) -> Tuple[KMeans, StandardScaler, Dict[int, str], Dict[str, Any]]:
    """
    Loads a persisted model artifact from disk.

    Parameters
    ----------
    version_dir : str
        Directory path of the model version.

    Returns
    -------
    Tuple[KMeans, StandardScaler, Dict[int, str], Dict[str, Any]]
        (model, scaler, cluster_map, metadata)
    """
    if not os.path.exists(version_dir):
        raise FileNotFoundError(f"Model version directory not found: '{version_dir}'")

    model_file = os.path.join(version_dir, "kmeans_model.joblib")
    scaler_file = os.path.join(version_dir, "scaler.joblib")
    meta_file = os.path.join(version_dir, "metadata.json")

    for req_file in (model_file, scaler_file, meta_file):
        if not os.path.exists(req_file):
            raise FileNotFoundError(f"Missing required artifact file: '{req_file}'")

    model: KMeans = joblib.load(model_file)
    scaler: StandardScaler = joblib.load(scaler_file)

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    raw_cluster_map = metadata.get("cluster_map", {})
    cluster_map: Dict[int, str] = {int(k): str(v) for k, v in raw_cluster_map.items()}

    return model, scaler, cluster_map, metadata


def predict_segment(
    model: KMeans,
    scaler: StandardScaler,
    cluster_map: Dict[int, str],
    rfm_data: Union[pd.DataFrame, Dict[str, float], List[Dict[str, float]]],
    feature_names: Optional[List[str]] = None,
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Predicts customer cluster index and persona label for new RFM records using a trained model.

    Consistent with the training pipeline, inputs are clipped at 0, log1p-transformed,
    scaled via StandardScaler, assigned cluster IDs via KMeans, and mapped to personas.

    Parameters
    ----------
    model : KMeans
        Fitted KMeans model.
    scaler : StandardScaler
        Fitted StandardScaler.
    cluster_map : Dict[int, str]
        Cluster index to persona label mapping.
    rfm_data : pd.DataFrame or Dict[str, float] or List[Dict[str, float]]
        Customer data with 'recency', 'frequency', and 'monetary' values.
    feature_names : List[str], optional
        Expected feature names. Defaults to ['recency', 'frequency', 'monetary'].

    Returns
    -------
    pd.DataFrame or Dict[str, Any]
        If a single dict is passed, returns a dict with predictions.
        If a DataFrame or list of dicts is passed, returns DataFrame with 'cluster' and 'segment'.
    """
    features = feature_names if feature_names is not None else DEFAULT_FEATURES

    # Single record dict prediction
    if isinstance(rfm_data, dict):
        for col in features:
            if col not in rfm_data:
                raise ValueError(f"Missing required RFM feature '{col}' in input dictionary.")

        raw_vector = np.array([float(rfm_data[col]) for col in features], dtype=float)
        clipped_vector = np.maximum(0.0, raw_vector)
        log_vector = np.log1p(clipped_vector).reshape(1, -1)
        scaled_vector = scaler.transform(log_vector)
        cluster_id = int(model.predict(scaled_vector)[0])
        segment_label = cluster_map.get(cluster_id, f"Cluster {cluster_id}")

        result: Dict[str, Any] = {
            "cluster": cluster_id,
            "segment": segment_label,
        }
        for col in features:
            result[col] = float(rfm_data[col])
        return result

    # List of dicts
    if isinstance(rfm_data, list):
        rfm_df = pd.DataFrame(rfm_data)
    elif isinstance(rfm_data, pd.DataFrame):
        rfm_df = rfm_data.copy()
    else:
        raise TypeError("rfm_data must be a pandas DataFrame, a dict, or a list of dicts.")

    for col in features:
        if col not in rfm_df.columns:
            raise ValueError(f"Feature column '{col}' missing from input DataFrame.")

    # Apply clipping, log1p, and standard scaling identically to training
    clipped_data = rfm_df[features].clip(lower=0.0).values
    log_transformed = np.log1p(clipped_data)
    scaled_data = scaler.transform(log_transformed)

    predicted_labels = model.predict(scaled_data)
    rfm_df["cluster"] = predicted_labels
    rfm_df["segment"] = rfm_df["cluster"].map(cluster_map).fillna("Unclassified")

    return rfm_df


def delete_model_version(version_dir: str) -> bool:
    """
    Safely deletes a model artifact version directory.

    Parameters
    ----------
    version_dir : str
        Directory to remove.

    Returns
    -------
    bool
        True if removed successfully, False otherwise.
    """
    if os.path.exists(version_dir) and os.path.isdir(version_dir):
        shutil.rmtree(version_dir)
        return True
    return False
