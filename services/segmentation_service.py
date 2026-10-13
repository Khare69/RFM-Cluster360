"""
Application & UI Orchestration Service Layer.

Bridges headless analytical core modules in `src/` with Streamlit's reactive UI
and caching infrastructure (@st.cache_data). This decouples pure domain algorithms
from Streamlit presentation mechanics while preserving responsive user experience.
"""

from datetime import date, datetime
import io
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import MAX_FILE_SIZE_MB, RANDOM_STATE
from src.clustering import (
    evaluate_clusters as _evaluate_clusters,
    preprocess_rfm as _preprocess_rfm,
    run_kmeans as _run_kmeans,
)
from src.ingestion import load_csv as _load_csv
from src.model_registry import save_model_artifact as _save_model_artifact
from src.rfm import calculate_rfm as _calculate_rfm


@st.cache_data(show_spinner=False)
def cached_load_csv(
    file_source: Union[str, bytes, io.BytesIO, io.StringIO, object],
    max_size_mb: int = MAX_FILE_SIZE_MB,
    dtype: Optional[Union[Dict, type]] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Cached wrapper for file ingestion with automatic encoding fallback and ID preservation.
    """
    return _load_csv(file_source=file_source, max_size_mb=max_size_mb, dtype=dtype)


@st.cache_data(show_spinner=False)
def cached_calculate_rfm(
    df: pd.DataFrame,
    snapshot_date: Optional[Union[str, date, datetime, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Cached wrapper for RFM metrics computation per unique customer.
    """
    return _calculate_rfm(df=df, snapshot_date=snapshot_date)


@st.cache_data(show_spinner=False)
def cached_preprocess_rfm(
    rfm_df: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Any, List[str]]:
    """
    Cached wrapper for log1p transformation and standardization of RFM features.
    """
    return _preprocess_rfm(rfm_df=rfm_df, features=features)


@st.cache_data(show_spinner=False)
def cached_evaluate_clusters(
    scaled_data: np.ndarray,
    k_range: Optional[range] = None,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Cached wrapper for multi-k cluster evaluation (elbow inertia and silhouette score).
    """
    return _evaluate_clusters(scaled_data=scaled_data, k_range=k_range, random_state=random_state)


def promote_model_to_registry(
    model: Any,
    scaler: Any,
    cluster_map: Dict[int, str],
    eval_metrics: Optional[Dict[str, Any]] = None,
    feature_names: Optional[List[str]] = None,
    customer_count: Optional[int] = None,
    notes: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> str:
    """
    Explicitly saves and promotes a trained session model into the MLOps model artifact registry.

    Parameters
    ----------
    model : KMeans or Pipeline
        Fitted Scikit-Learn KMeans or unified Pipeline.
    scaler : StandardScaler, optional
        Fitted StandardScaler.
    cluster_map : Dict[int, str]
        Cluster index to persona mapping.
    eval_metrics : Dict[str, Any], optional
        Evaluation diagnostics.
    feature_names : List[str], optional
        Feature column names.
    customer_count : int, optional
        Number of customer records.
    notes : str, optional
        Descriptive tag or commit notes.
    save_dir : str, optional
        Custom registry directory.

    Returns
    -------
    str
        Path to the newly created version directory in the registry.
    """
    return _save_model_artifact(
        model=model,
        scaler=scaler,
        cluster_map=cluster_map,
        eval_metrics=eval_metrics,
        feature_names=feature_names,
        customer_count=customer_count,
        notes=notes,
        save_dir=save_dir,
    )
