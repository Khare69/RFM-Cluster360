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
from src.clv import (
    calculate_customer_clv as _calculate_customer_clv,
    fit_bgnbd as _fit_bgnbd,
    fit_gamma_gamma as _fit_gamma_gamma,
    prepare_clv_summary as _prepare_clv_summary,
)
from src.clustering import (
    align_centroids as _align_centroids,
    evaluate_clusters as _evaluate_clusters,
    isolate_whales as _isolate_whales,
    preprocess_rfm as _preprocess_rfm,
)
from src.ingestion import load_csv as _load_csv
from src.logger import setup_logger
from src.model_registry import save_model_artifact as _save_model_artifact
from src.reconciler import reconcile_transactions as _reconcile_transactions
from src.rfm import calculate_rfm as _calculate_rfm

logger = setup_logger("services.segmentation_service")


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
def cached_reconcile_transactions(
    df: pd.DataFrame,
    lookback_window_days: int = 180,
) -> Tuple[pd.DataFrame, Any]:
    """
    Cached wrapper for two-pass net-settlement transaction reconciliation.
    """
    logger.info("Executing two-pass transaction reconciliation (lookback=%d days)", lookback_window_days)
    return _reconcile_transactions(df=df, lookback_window_days=lookback_window_days)


@st.cache_data(show_spinner=False)
def cached_calculate_rfm(
    df: pd.DataFrame,
    snapshot_date: Optional[Union[str, date, datetime, pd.Timestamp]] = None,
    lookback_days: Optional[int] = None,
    use_average_monetary: bool = False,
) -> pd.DataFrame:
    """
    Cached wrapper for RFM metrics computation per unique customer.
    """
    logger.info("Computing customer RFM (lookback=%s, avg_monetary=%s)", lookback_days, use_average_monetary)
    return _calculate_rfm(
        df=df,
        snapshot_date=snapshot_date,
        lookback_days=lookback_days,
        use_average_monetary=use_average_monetary,
    )


@st.cache_data(show_spinner=False)
def cached_preprocess_rfm(
    rfm_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    scaler_type: str = "standard",
) -> Tuple[np.ndarray, Any, List[str]]:
    """
    Cached wrapper for log1p transformation and standardization of RFM features.
    """
    logger.info("Preprocessing RFM matrix with scaler_type=%s", scaler_type)
    return _preprocess_rfm(rfm_df=rfm_df, features=features, scaler_type=scaler_type)


@st.cache_data(show_spinner=False)
def cached_calculate_clv(
    transactions_df: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
    horizon_days: float = 180.0,
) -> pd.DataFrame:
    """
    Cached wrapper for probabilistic Customer Lifetime Value modeling (BG/NBD + Gamma-Gamma).
    """
    logger.info("Fitting probabilistic CLV models (BG/NBD + Gamma-Gamma) for horizon=%.1f days", horizon_days)
    summary_df = _prepare_clv_summary(transactions_df, snapshot_date=snapshot_date)
    bgnbd_params = _fit_bgnbd(summary_df)
    gg_params = _fit_gamma_gamma(summary_df)
    return _calculate_customer_clv(summary_df, bgnbd_params, gg_params, horizon_days=horizon_days)


def isolate_whales_service(
    rfm_df: pd.DataFrame,
    quantile: float = 0.995,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Two-stage hybrid isolation wrapper to segregate extreme upper-tail accounts from retail core.
    """
    return _isolate_whales(rfm_df=rfm_df, quantile=quantile)


def align_centroids_service(
    centroids: np.ndarray,
    anchor_centroids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Centroid alignment wrapper using Hungarian matching to prevent label switching.
    """
    return _align_centroids(centroids=centroids, anchor_centroids=anchor_centroids)


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
