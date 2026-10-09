"""
Unit tests for src/model_registry.py.
Verifies serialization, deserialization, versioning, metadata tracking,
artifact deletion, and real-time customer segmentation inference.
"""

import json
import os
import shutil
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.model_registry import (
    delete_model_version,
    list_model_versions,
    load_model_artifact,
    predict_segment,
    save_model_artifact,
)


@pytest.fixture
def trained_ml_bundle():
    """Generates a fitted KMeans model, fitted StandardScaler, cluster map, and sample data."""
    np.random.seed(42)
    # 30 synthetic customer RFM samples
    r = np.random.uniform(5, 300, 30)
    f = np.random.uniform(1, 20, 30)
    m = np.random.uniform(20, 2000, 30)
    raw_matrix = np.column_stack([r, f, m])

    log_transformed = np.log1p(raw_matrix)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(log_transformed)

    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)

    cluster_map = {0: "Champions", 1: "Loyal Customers", 2: "At Risk"}
    eval_metrics = {
        "k_values": [2, 3, 4],
        "silhouette_scores": [0.3812, 0.4521, 0.4019],
        "best_k": 3,
        "best_silhouette": 0.4521,
        "inertias": [120.5, 78.4, 60.1],
    }

    df = pd.DataFrame(raw_matrix, columns=["recency", "frequency", "monetary"])
    df["customer_id"] = [f"CUST_{i}" for i in range(len(df))]

    return {
        "model": kmeans,
        "scaler": scaler,
        "cluster_map": cluster_map,
        "eval_metrics": eval_metrics,
        "df": df,
    }


def test_save_and_load_roundtrip(tmp_path, trained_ml_bundle):
    """Verifies that model artifacts can be serialized to disk and loaded back with full fidelity."""
    save_dir = str(tmp_path / "models")
    bundle = trained_ml_bundle

    version_dir = save_model_artifact(
        model=bundle["model"],
        scaler=bundle["scaler"],
        cluster_map=bundle["cluster_map"],
        eval_metrics=bundle["eval_metrics"],
        feature_names=["recency", "frequency", "monetary"],
        customer_count=len(bundle["df"]),
        notes="Test baseline model artifact",
        save_dir=save_dir,
    )

    assert os.path.exists(version_dir)
    assert os.path.exists(os.path.join(version_dir, "kmeans_model.joblib"))
    assert os.path.exists(os.path.join(version_dir, "scaler.joblib"))
    assert os.path.exists(os.path.join(version_dir, "metadata.json"))

    # Load back
    loaded_model, loaded_scaler, loaded_cmap, loaded_meta = load_model_artifact(version_dir)

    assert isinstance(loaded_model, KMeans)
    assert loaded_model.n_clusters == 3
    np.testing.assert_array_almost_equal(loaded_model.cluster_centers_, bundle["model"].cluster_centers_)

    assert isinstance(loaded_scaler, StandardScaler)
    np.testing.assert_array_almost_equal(loaded_scaler.mean_, bundle["scaler"].mean_)
    np.testing.assert_array_almost_equal(loaded_scaler.scale_, bundle["scaler"].scale_)

    assert loaded_cmap == bundle["cluster_map"]
    assert loaded_meta["k"] == 3
    assert loaded_meta["silhouette_score"] == 0.4521
    assert loaded_meta["customer_count"] == 30
    assert loaded_meta["notes"] == "Test baseline model artifact"


def test_list_model_versions(tmp_path, trained_ml_bundle):
    """Verifies that model versions are listed properly, sorted newest first."""
    save_dir = str(tmp_path / "registry")
    bundle = trained_ml_bundle

    # Initially empty
    assert list_model_versions(save_dir) == []

    # Save first version
    v1 = save_model_artifact(
        model=bundle["model"],
        scaler=bundle["scaler"],
        cluster_map=bundle["cluster_map"],
        save_dir=save_dir,
        notes="Version 1",
    )

    # Save second version with different note
    v2 = save_model_artifact(
        model=bundle["model"],
        scaler=bundle["scaler"],
        cluster_map=bundle["cluster_map"],
        save_dir=save_dir,
        notes="Version 2",
    )

    versions = list_model_versions(save_dir)
    assert len(versions) == 2
    version_ids = [v["version_id"] for v in versions]
    assert os.path.basename(v2) in version_ids
    assert os.path.basename(v1) in version_ids
    assert versions[0]["version_id"] == os.path.basename(v2)


def test_predict_segment_single_record(trained_ml_bundle):
    """Verifies real-time prediction for a single dictionary record."""
    bundle = trained_ml_bundle
    sample_input = {"recency": 12.0, "frequency": 5.0, "monetary": 350.0}

    pred = predict_segment(
        model=bundle["model"],
        scaler=bundle["scaler"],
        cluster_map=bundle["cluster_map"],
        rfm_data=sample_input,
    )

    assert isinstance(pred, dict)
    assert "cluster" in pred
    assert "segment" in pred
    assert pred["cluster"] in [0, 1, 2]
    assert pred["segment"] in ["Champions", "Loyal Customers", "At Risk"]
    assert pred["recency"] == 12.0


def test_predict_segment_dataframe(trained_ml_bundle):
    """Verifies batch prediction for a pandas DataFrame."""
    bundle = trained_ml_bundle
    df = bundle["df"].copy()

    result_df = predict_segment(
        model=bundle["model"],
        scaler=bundle["scaler"],
        cluster_map=bundle["cluster_map"],
        rfm_data=df,
    )

    assert isinstance(result_df, pd.DataFrame)
    assert "cluster" in result_df.columns
    assert "segment" in result_df.columns
    assert len(result_df) == len(df)
    assert set(result_df["cluster"].unique()).issubset({0, 1, 2})
    assert set(result_df["segment"].unique()).issubset({"Champions", "Loyal Customers", "At Risk"})


def test_predict_segment_missing_features(trained_ml_bundle):
    """Verifies that missing RFM features raise a descriptive ValueError."""
    bundle = trained_ml_bundle
    invalid_dict = {"recency": 10.0, "monetary": 100.0}  # missing 'frequency'

    with pytest.raises(ValueError, match="Missing required RFM feature 'frequency'"):
        predict_segment(
            model=bundle["model"],
            scaler=bundle["scaler"],
            cluster_map=bundle["cluster_map"],
            rfm_data=invalid_dict,
        )

    invalid_df = pd.DataFrame([invalid_dict])
    with pytest.raises(ValueError, match="Feature column 'frequency' missing"):
        predict_segment(
            model=bundle["model"],
            scaler=bundle["scaler"],
            cluster_map=bundle["cluster_map"],
            rfm_data=invalid_df,
        )


def test_load_nonexistent_or_corrupt_artifact(tmp_path):
    """Verifies FileNotFoundError on missing directory or missing artifact components."""
    with pytest.raises(FileNotFoundError, match="Model version directory not found"):
        load_model_artifact(str(tmp_path / "nonexistent"))

    # Create directory missing required joblib files
    empty_dir = tmp_path / "incomplete_v1"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Missing required artifact file"):
        load_model_artifact(str(empty_dir))


def test_delete_model_version(tmp_path, trained_ml_bundle):
    """Verifies safe deletion of artifact directories."""
    save_dir = str(tmp_path / "del_models")
    bundle = trained_ml_bundle

    version_dir = save_model_artifact(
        model=bundle["model"],
        scaler=bundle["scaler"],
        cluster_map=bundle["cluster_map"],
        save_dir=save_dir,
    )
    assert os.path.exists(version_dir)

    success = delete_model_version(version_dir)
    assert success is True
    assert not os.path.exists(version_dir)

    # Deleting non-existent returns False
    assert delete_model_version(version_dir) is False


def test_save_model_type_validation(trained_ml_bundle):
    """Verifies that invalid object types raise TypeError."""
    bundle = trained_ml_bundle
    with pytest.raises(TypeError, match="model must be an instance of sklearn.cluster.KMeans"):
        save_model_artifact(
            model="not_a_model",  # type: ignore
            scaler=bundle["scaler"],
            cluster_map=bundle["cluster_map"],
        )

    with pytest.raises(TypeError, match="scaler must be an instance of sklearn.preprocessing.StandardScaler"):
        save_model_artifact(
            model=bundle["model"],
            scaler="not_a_scaler",  # type: ignore
            cluster_map=bundle["cluster_map"],
        )
