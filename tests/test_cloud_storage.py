"""
Unit tests for S3 cloud storage integration, Parquet serialization, and local mock storage.
"""

import io
import os
import boto3
from moto import mock_aws
import pandas as pd
import pytest

from src.cloud_storage import (
    get_mock_s3_path,
    is_mock_mode,
    load_from_s3,
    upload_to_s3,
)
from src.export import to_csv_bytes, to_parquet_bytes


@pytest.fixture
def sample_rfm_df():
    """Provides a realistic sample DataFrame for testing serialization and cloud storage."""
    return pd.DataFrame(
        {
            "customer_id": ["1001", "1002", "1003"],
            "recency": [5, 45, 120],
            "frequency": [10, 3, 1],
            "monetary": [540.50, 120.00, 25.00],
            "cluster": [0, 1, 2],
            "segment": ["Champions", "Promising", "Lost / Dormant"],
            "first_purchase": ["2023-01-10", "2023-03-15", "2023-05-20"],
            "last_purchase": ["2023-12-25", "2023-11-15", "2023-08-30"],
        }
    )


def test_to_parquet_bytes_roundtrip(sample_rfm_df):
    """Verifies DataFrame serializes to valid Apache Parquet binary and round-trips correctly."""
    parquet_bytes = to_parquet_bytes(sample_rfm_df)

    assert isinstance(parquet_bytes, bytes)
    assert len(parquet_bytes) > 0
    # Parquet files begin with magic 4 bytes 'PAR1'
    assert parquet_bytes.startswith(b"PAR1")

    # Read back via pyarrow engine
    loaded_df = pd.read_parquet(io.BytesIO(parquet_bytes))
    assert len(loaded_df) == len(sample_rfm_df)
    assert list(loaded_df.columns) == list(sample_rfm_df.columns)
    assert loaded_df["customer_id"].tolist() == ["1001", "1002", "1003"]
    assert loaded_df["monetary"].tolist() == [540.50, 120.00, 25.00]


def test_to_parquet_bytes_columns_filter(sample_rfm_df):
    """Verifies that selecting specific columns filters the exported Parquet file."""
    selected_cols = ["customer_id", "segment", "monetary"]
    parquet_bytes = to_parquet_bytes(sample_rfm_df, columns=selected_cols)

    loaded_df = pd.read_parquet(io.BytesIO(parquet_bytes))
    assert list(loaded_df.columns) == selected_cols
    assert len(loaded_df) == 3


@mock_aws
def test_load_from_s3_csv_with_moto(sample_rfm_df):
    """Tests loading a CSV file from an S3 bucket mocked via moto."""
    bucket_name = "test-retail-datalake"
    object_key = "raw/transactions.csv"
    region = "us-east-1"

    s3 = boto3.client("s3", region_name=region)
    s3.create_bucket(Bucket=bucket_name)

    csv_data = sample_rfm_df.to_csv(index=False).encode("utf-8")
    s3.put_object(Bucket=bucket_name, Key=object_key, Body=csv_data)

    df_loaded, err = load_from_s3(
        bucket=bucket_name,
        key=object_key,
        region_name=region,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )

    assert err is None
    assert df_loaded is not None
    assert len(df_loaded) == 3
    assert "customer_id" in df_loaded.columns
    assert "monetary" in df_loaded.columns


@mock_aws
def test_load_from_s3_parquet_with_moto(sample_rfm_df):
    """Tests loading an Apache Parquet file from an S3 bucket mocked via moto."""
    bucket_name = "test-parquet-datalake"
    object_key = "analytics/customer_segments.parquet"
    region = "us-east-1"

    s3 = boto3.client("s3", region_name=region)
    s3.create_bucket(Bucket=bucket_name)

    parquet_data = to_parquet_bytes(sample_rfm_df)
    s3.put_object(Bucket=bucket_name, Key=object_key, Body=parquet_data)

    df_loaded, err = load_from_s3(
        bucket=bucket_name,
        key=object_key,
        region_name=region,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )

    assert err is None
    assert df_loaded is not None
    assert len(df_loaded) == 3
    assert df_loaded["customer_id"].tolist() == ["1001", "1002", "1003"]
    assert df_loaded["segment"].tolist() == ["Champions", "Promising", "Lost / Dormant"]


@mock_aws
def test_upload_to_s3_parquet_and_csv_with_moto(sample_rfm_df):
    """Tests exporting DataFrame directly to S3 as Parquet and CSV via moto."""
    bucket_name = "export-target-bucket"
    region = "us-east-1"

    s3 = boto3.client("s3", region_name=region)
    s3.create_bucket(Bucket=bucket_name)

    # Test Parquet export
    pq_key = "exports/2026/segments.parquet"
    success, uri = upload_to_s3(
        df=sample_rfm_df,
        bucket=bucket_name,
        key=pq_key,
        format="parquet",
        region_name=region,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    assert success is True
    assert uri == f"s3://{bucket_name}/{pq_key}"

    # Verify object in bucket and read back via load_from_s3
    df_read_pq, err_pq = load_from_s3(
        bucket=bucket_name,
        key=pq_key,
        region_name=region,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    assert err_pq is None
    assert len(df_read_pq) == 3

    # Test CSV export
    csv_key = "exports/2026/segments.csv"
    success_csv, uri_csv = upload_to_s3(
        df=sample_rfm_df,
        bucket=bucket_name,
        key=csv_key,
        format="csv",
        region_name=region,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    assert success_csv is True
    assert uri_csv == f"s3://{bucket_name}/{csv_key}"

    df_read_csv, err_csv = load_from_s3(
        bucket=bucket_name,
        key=csv_key,
        region_name=region,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    assert err_csv is None
    assert len(df_read_csv) == 3


@mock_aws
def test_load_from_s3_nonexistent_key_moto():
    """Verifies that attempting to load a missing object returns a descriptive S3 error."""
    bucket_name = "test-empty-bucket"
    region = "us-east-1"

    s3 = boto3.client("s3", region_name=region)
    s3.create_bucket(Bucket=bucket_name)

    df_loaded, err = load_from_s3(
        bucket=bucket_name,
        key="nonexistent_folder/missing.csv",
        region_name=region,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )

    assert df_loaded is None
    assert err is not None
    assert "NoSuchKey" in err


def test_local_mock_mode_load_and_upload(sample_rfm_df, tmp_path, monkeypatch):
    """Verifies local mock mode loads and exports from data/mock_s3 directory without network calls."""
    monkeypatch.setenv("USE_LOCAL_MOCK", "true")
    monkeypatch.setenv("MOCK_S3_DIR", str(tmp_path))

    bucket = "mock-retail"
    key_csv = "transactions.csv"
    key_pq = "output.parquet"

    # Pre-populate mock directory
    mock_csv_path = tmp_path / bucket / key_csv
    mock_csv_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rfm_df.to_csv(mock_csv_path, index=False)

    assert is_mock_mode() is True

    # Test load from mock
    df_loaded, err = load_from_s3(bucket=bucket, key=key_csv, mock_dir=str(tmp_path))
    assert err is None
    assert df_loaded is not None
    assert len(df_loaded) == len(sample_rfm_df)

    # Test upload to mock
    success, uri = upload_to_s3(
        df=sample_rfm_df,
        bucket=bucket,
        key=key_pq,
        format="parquet",
        mock_dir=str(tmp_path),
    )
    assert success is True
    assert uri == f"mock://{bucket}/{key_pq}"

    # Verify mock file was physically created
    mock_pq_path = tmp_path / bucket / key_pq
    assert mock_pq_path.exists()
    assert mock_pq_path.stat().st_size > 0

    # Load back the mock parquet file
    df_reloaded, err2 = load_from_s3(bucket=bucket, key=key_pq, mock_dir=str(tmp_path))
    assert err2 is None
    assert len(df_reloaded) == len(sample_rfm_df)


def test_load_from_s3_input_validation():
    """Verifies validation guards against empty or whitespace bucket and key inputs."""
    df1, err1 = load_from_s3("", "")
    assert df1 is None
    assert "Both bucket name and object key must be provided." in err1

    df2, err2 = load_from_s3("bucket-only", "")
    assert df2 is None
    assert "Both bucket name and object key must be provided." in err2


def test_upload_to_s3_input_validation(sample_rfm_df):
    """Verifies validation guards on upload parameters (invalid df, empty df, unsupported format)."""
    # Non-dataframe input
    ok1, err1 = upload_to_s3("not_a_df", "b", "k")
    assert ok1 is False
    assert "Input must be a valid pandas DataFrame" in err1

    # Empty dataframe
    ok2, err2 = upload_to_s3(pd.DataFrame(), "b", "k")
    assert ok2 is False
    assert "Cannot upload an empty DataFrame" in err2

    # Empty bucket/key
    ok3, err3 = upload_to_s3(sample_rfm_df, "", "key")
    assert ok3 is False
    assert "Both bucket name and object key must be provided." in err3

    # Unsupported format
    ok4, err4 = upload_to_s3(sample_rfm_df, "b", "k", format="xml")
    assert ok4 is False
    assert "Unsupported format 'xml'" in err4
