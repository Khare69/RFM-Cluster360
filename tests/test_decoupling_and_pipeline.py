"""
Unit tests for Phase 11: Architectural Decoupling & Unified Pipeline Robustness.
Verifies that:
1. `src/` modules are completely decoupled from Streamlit (zero Streamlit imports or decorators).
2. Pure headless analytical execution of RFM calculation, clustering, and ingestion.
3. Unified Scikit-Learn Pipeline creation, fitting, serialization, and inference.
4. Parity between Pipeline prediction and traditional scaler + KMeans prediction.
5. Legacy fallback support when loading artifacts without `pipeline.joblib`.
6. UI service orchestration layer functionality.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.clustering import (
    create_rfm_pipeline,
    evaluate_clusters,
    fit_rfm_pipeline,
    preprocess_rfm,
    run_kmeans,
)
from src.model_registry import (
    load_pipeline_artifact,
    predict_segment,
    save_model_artifact,
)
from src.rfm import calculate_rfm, default_snapshot_date
from services.segmentation_service import (
    cached_load_csv,
    promote_model_to_registry,
)


def test_src_modules_have_no_streamlit_dependencies():
    """
    Scans every Python file in the `src/` directory to strictly enforce that
    no file imports `streamlit` or uses `@st.cache_data`.
    """
    src_dir = Path(__file__).resolve().parent.parent / "src"
    py_files = list(src_dir.glob("*.py"))
    assert len(py_files) > 0, "No python files found in src/ directory"

    violating_files = []
    import re
    for py_file in py_files:
        content = py_file.read_text(encoding="utf-8")
        if re.search(r"^\s*(import\s+streamlit|from\s+streamlit|@st\.)", content, re.MULTILINE):
            violating_files.append(py_file.name)

    assert violating_files == [], (
        f"Streamlit imports or decorators found in core analytics modules: {violating_files}. "
        "Core logic must be completely decoupled from UI framework."
    )


def test_headless_pure_python_pipeline_execution():
    """
    Verifies that the entire analytical pipeline can run headlessly
    without any Streamlit session state or UI runtime.
    """
    # 1. Synthetic transaction data
    data = {
        "customer_id": ["C1", "C1", "C2", "C3", "C3", "C3"],
        "invoice_no": ["101", "102", "103", "104", "105", "106"],
        "invoice_date": [
            "2023-01-01",
            "2023-01-05",
            "2023-02-01",
            "2023-03-01",
            "2023-03-02",
            "2023-03-05",
        ],
        "quantity": [2, 1, 5, 1, 2, 3],
        "unit_price": [10.0, 20.0, 15.0, 100.0, 50.0, 20.0],
    }
    df = pd.DataFrame(data)

    # 2. Ingestion & Snapshot
    snap_dt = default_snapshot_date(df)
    assert isinstance(snap_dt, pd.Timestamp)

    # 3. Headless RFM calculation
    rfm_df = calculate_rfm(df, snapshot_date=snap_dt)
    assert len(rfm_df) == 3
    assert set(rfm_df.columns).issuperset({"customer_id", "recency", "frequency", "monetary"})

    # 4. Multi-sample synthetic for clustering
    np.random.seed(42)
    extended_rfm = pd.DataFrame({
        "customer_id": [f"CUST_{i}" for i in range(25)],
        "recency": np.random.uniform(1, 100, 25),
        "frequency": np.random.uniform(1, 15, 25),
        "monetary": np.random.uniform(20, 1000, 25),
    })

    scaled_data, scaler, features = preprocess_rfm(extended_rfm)
    assert scaled_data.shape == (25, 3)

    eval_res = evaluate_clusters(scaled_data, k_range=range(2, 5))
    assert eval_res["best_k"] in [2, 3, 4]

    labels, centers, model = run_kmeans(scaled_data, k=eval_res["best_k"])
    assert len(labels) == 25
    assert isinstance(model, KMeans)


def test_create_and_fit_rfm_pipeline():
    """
    Verifies that create_rfm_pipeline constructs a unified Scikit-Learn Pipeline
    and fit_rfm_pipeline trains it end-to-end on raw RFM features.
    """
    np.random.seed(42)
    rfm_df = pd.DataFrame({
        "customer_id": [f"C_{i}" for i in range(30)],
        "recency": np.random.uniform(2, 200, 30),
        "frequency": np.random.uniform(1, 20, 30),
        "monetary": np.random.uniform(10, 2000, 30),
    })

    pipeline = create_rfm_pipeline(k=3, random_state=42)
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps.keys()) == ["log1p", "scaler", "kmeans"]

    # Fit pipeline
    fitted_pipe, labels = fit_rfm_pipeline(rfm_df, k=3, random_state=42)
    assert len(labels) == 30
    assert set(np.unique(labels)).issubset({0, 1, 2})

    # Predict with pipeline on new raw customer
    new_data = np.array([[10.0, 5.0, 500.0]])
    pred_label = fitted_pipe.predict(new_data)
    assert pred_label[0] in [0, 1, 2]


def test_unified_pipeline_artifact_serialization_roundtrip(tmp_path):
    """
    Verifies that saving a model artifact automatically produces a unified `pipeline.joblib`,
    which can be loaded back via `load_pipeline_artifact` and produces identical predictions.
    """
    save_dir = str(tmp_path / "unified_registry")

    np.random.seed(42)
    rfm_df = pd.DataFrame({
        "customer_id": [f"C_{i}" for i in range(30)],
        "recency": np.random.uniform(2, 200, 30),
        "frequency": np.random.uniform(1, 20, 30),
        "monetary": np.random.uniform(10, 2000, 30),
    })

    pipe, _ = fit_rfm_pipeline(rfm_df, k=3, random_state=42)
    cluster_map = {0: "Champions", 1: "Loyal Customers", 2: "At Risk"}

    # Save artifact using the unified pipeline
    version_dir = save_model_artifact(
        model=pipe,
        cluster_map=cluster_map,
        save_dir=save_dir,
        notes="Unified Pipeline Test Artifact",
    )

    assert os.path.exists(version_dir)
    assert os.path.exists(os.path.join(version_dir, "pipeline.joblib"))
    assert os.path.exists(os.path.join(version_dir, "kmeans_model.joblib"))
    assert os.path.exists(os.path.join(version_dir, "scaler.joblib"))
    assert os.path.exists(os.path.join(version_dir, "metadata.json"))

    # Load back using load_pipeline_artifact
    loaded_pipe, loaded_cmap, loaded_meta = load_pipeline_artifact(version_dir)
    assert isinstance(loaded_pipe, Pipeline)
    assert loaded_cmap == cluster_map
    assert loaded_meta["has_pipeline"] is True
    assert loaded_meta["k"] == 3

    # Verify identical predictions on holdout customers
    test_customers = pd.DataFrame({
        "recency": [5.0, 150.0, 30.0],
        "frequency": [12.0, 1.0, 6.0],
        "monetary": [1500.0, 25.0, 400.0],
    })

    pipe_preds = loaded_pipe.predict(test_customers[["recency", "frequency", "monetary"]].values)
    pred_df = predict_segment(
        model=loaded_pipe,
        cluster_map=loaded_cmap,
        rfm_data=test_customers,
    )

    assert np.array_equal(pred_df["cluster"].values, pipe_preds)
    assert pred_df["segment"].iloc[0] in ["Champions", "Loyal Customers", "At Risk"]


def test_pipeline_parity_with_step_by_step_inference():
    """
    Verifies that unified Pipeline prediction produces EXACT parity with
    the manual step-by-step: clip -> log1p -> StandardScaler -> KMeans.
    """
    np.random.seed(42)
    raw_x = np.random.uniform(5, 500, (40, 3))
    rfm_df = pd.DataFrame(raw_x, columns=["recency", "frequency", "monetary"])

    # Method 1: Manual step-by-step
    scaled_data, scaler, _ = preprocess_rfm(rfm_df)
    labels, _, kmeans = run_kmeans(scaled_data, k=3, random_state=42)

    # Method 2: Pipeline created with the same fitted scaler and kmeans
    pipe = create_rfm_pipeline(k=3, random_state=42, scaler=scaler, kmeans_model=kmeans)

    test_samples = np.array([
        [10.0, 2.0, 100.0],
        [200.0, 1.0, 30.0],
        [15.0, 10.0, 1200.0],
    ])

    # Predict via manual steps
    manual_scaled = scaler.transform(np.log1p(np.maximum(0.0, test_samples)))
    manual_preds = kmeans.predict(manual_scaled)

    # Predict via pipeline
    pipe_preds = pipe.predict(test_samples)

    np.testing.assert_array_equal(pipe_preds, manual_preds)


def test_legacy_artifact_fallback_without_pipeline_joblib(tmp_path):
    """
    Verifies that load_pipeline_artifact gracefully reconstructs a unified Pipeline
    when loading legacy model directories that only contain separate joblib files.
    """
    legacy_dir = tmp_path / "legacy_v1"
    legacy_dir.mkdir()

    # Create dummy KMeans and StandardScaler
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    scaler = StandardScaler()
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    scaler.fit(X)
    kmeans.fit(scaler.transform(X))

    import joblib
    import json
    joblib.dump(kmeans, str(legacy_dir / "kmeans_model.joblib"))
    joblib.dump(scaler, str(legacy_dir / "scaler.joblib"))

    meta = {
        "version_id": "legacy_v1",
        "k": 2,
        "cluster_map": {"0": "Segment A", "1": "Segment B"},
    }
    with open(str(legacy_dir / "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # load_pipeline_artifact should fall back and construct Pipeline
    reconstructed_pipe, cmap, _ = load_pipeline_artifact(str(legacy_dir))
    assert isinstance(reconstructed_pipe, Pipeline)
    assert cmap == {0: "Segment A", 1: "Segment B"}

    preds = reconstructed_pipe.predict(np.array([[2.0, 3.0, 4.0]]))
    assert preds[0] in [0, 1]


def test_segmentation_service_orchestration(tmp_path):
    """
    Verifies that services.segmentation_service functions can be invoked
    and promote_model_to_registry successfully writes to disk.
    """
    save_dir = str(tmp_path / "promo_registry")

    # Ingestion via service
    csv_bytes = b"customer_id,invoice_no,invoice_date,quantity,unit_price\nC1,101,2023-01-01,2,10.0"
    df, err = cached_load_csv(csv_bytes)
    assert err is None
    assert len(df) == 1

    # Promotion via service
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    scaler = StandardScaler()
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    scaler.fit(X)
    kmeans.fit(scaler.transform(X))

    saved_path = promote_model_to_registry(
        model=kmeans,
        scaler=scaler,
        cluster_map={0: "Tier 1", 1: "Tier 2"},
        customer_count=3,
        notes="Explicit Promotion Test",
        save_dir=save_dir,
    )

    assert os.path.exists(saved_path)
    assert os.path.exists(os.path.join(saved_path, "pipeline.joblib"))
