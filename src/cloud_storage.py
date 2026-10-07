"""
Cloud storage ingestion and export module.
Supports AWS S3 object reading and writing (CSV and Apache Parquet)
with an offline local mock mode for development, testing, and demos.
"""

import io
import os
from typing import Optional, Tuple
import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError
import pandas as pd

from config.settings import (
    AWS_ACCESS_KEY_ID,
    AWS_DEFAULT_REGION,
    AWS_SECRET_ACCESS_KEY,
    MOCK_S3_DIR,
    USE_LOCAL_MOCK,
)
from src.export import to_csv_bytes, to_parquet_bytes
from src.ingestion import load_csv


def is_mock_mode() -> bool:
    """
    Checks if local mock mode is enabled via environment or configuration.

    Returns
    -------
    bool
        True if mock mode is active, False otherwise.
    """
    env_mock = os.getenv("USE_LOCAL_MOCK", "").lower()
    if env_mock in ("true", "1", "yes"):
        return True
    if env_mock in ("false", "0", "no"):
        return False
    return USE_LOCAL_MOCK


def get_mock_s3_path(bucket: str, key: str, mock_dir: Optional[str] = None) -> str:
    """
    Resolves the local filesystem path for a mock S3 bucket and object key.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    key : str
        S3 object key / relative path.
    mock_dir : str, optional
        Custom mock root directory; defaults to MOCK_S3_DIR.

    Returns
    -------
    str
        Absolute local file path.
    """
    base = mock_dir or os.getenv("MOCK_S3_DIR", MOCK_S3_DIR)
    clean_bucket = bucket.replace("s3://", "").strip("/\\")
    clean_key = key.lstrip("/\\")
    return os.path.abspath(os.path.join(base, clean_bucket, clean_key))


def get_s3_client(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    region_name: Optional[str] = None,
    endpoint_url: Optional[str] = None,
):
    """
    Initializes and returns a configured boto3 S3 client.

    Parameters
    ----------
    aws_access_key_id : str, optional
        Explicit AWS access key; falls back to environment.
    aws_secret_access_key : str, optional
        Explicit AWS secret access key; falls back to environment.
    region_name : str, optional
        Explicit AWS region; falls back to environment or us-east-1.
    endpoint_url : str, optional
        Custom S3 endpoint URL (e.g. MinIO, LocalStack).

    Returns
    -------
    boto3.client
        Configured boto3 S3 client.
    """
    key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID", AWS_ACCESS_KEY_ID) or None
    secret_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY", AWS_SECRET_ACCESS_KEY) or None
    region = region_name or os.getenv("AWS_DEFAULT_REGION", AWS_DEFAULT_REGION) or "us-east-1"

    kwargs = {"region_name": region}
    if key_id and secret_key:
        kwargs["aws_access_key_id"] = key_id
        kwargs["aws_secret_access_key"] = secret_key
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    return boto3.client("s3", **kwargs)


def load_from_s3(
    bucket: str,
    key: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    region_name: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    mock_dir: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Downloads and parses a CSV or Parquet file from an S3 bucket or local mock storage.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    key : str
        S3 object key / file path.
    aws_access_key_id : str, optional
        AWS Access Key ID override.
    aws_secret_access_key : str, optional
        AWS Secret Access Key override.
    region_name : str, optional
        AWS Region override.
    endpoint_url : str, optional
        Optional custom S3 endpoint URL.
    mock_dir : str, optional
        Optional custom local mock root directory.

    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[str]]
        (DataFrame, None) on success, or (None, error_message) on failure.
    """
    clean_bucket = bucket.replace("s3://", "").strip("/\\") if bucket else ""
    clean_key = key.lstrip("/\\") if key else ""

    if not clean_bucket or not clean_key:
        return None, "Both bucket name and object key must be provided."

    data_bytes: Optional[bytes] = None

    # Check if local mock mode is forced
    if is_mock_mode():
        mock_file_path = get_mock_s3_path(clean_bucket, clean_key, mock_dir=mock_dir)
        if not os.path.exists(mock_file_path):
            return None, f"Local mock file not found: {mock_file_path}"
        try:
            with open(mock_file_path, "rb") as f:
                data_bytes = f.read()
        except Exception as e:
            return None, f"Failed to read local mock file: {str(e)}"
    else:
        # Attempt download from S3 via boto3 (or moto)
        try:
            s3_client = get_s3_client(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name,
                endpoint_url=endpoint_url,
            )
            response = s3_client.get_object(Bucket=clean_bucket, Key=clean_key)
            data_bytes = response["Body"].read()
        except NoCredentialsError:
            # Fall back to local mock storage if file exists on disk
            mock_file_path = get_mock_s3_path(clean_bucket, clean_key, mock_dir=mock_dir)
            if os.path.exists(mock_file_path):
                try:
                    with open(mock_file_path, "rb") as f:
                        data_bytes = f.read()
                except Exception as e:
                    return None, f"Failed to read local fallback file: {str(e)}"
            else:
                return (
                    None,
                    "AWS credentials not configured. Provide AWS_ACCESS_KEY_ID and "
                    "AWS_SECRET_ACCESS_KEY in .env, or enable USE_LOCAL_MOCK=true to demo with local mock storage.",
                )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "Unknown")
            msg = e.response.get("Error", {}).get("Message", str(e))
            return None, f"S3 Error [{code}]: {msg}"
        except EndpointConnectionError as e:
            return None, f"Could not connect to S3 endpoint: {str(e)}"
        except Exception as e:
            return None, f"Failed to download s3://{clean_bucket}/{clean_key}: {str(e)}"

    if data_bytes is None or len(data_bytes) == 0:
        return None, f"File s3://{clean_bucket}/{clean_key} is empty."

    # Parse based on file type (Parquet vs CSV)
    is_parquet = clean_key.lower().endswith((".parquet", ".pq")) or data_bytes.startswith(b"PAR1")

    if is_parquet:
        try:
            df = pd.read_parquet(io.BytesIO(data_bytes))
            return df, None
        except Exception as e:
            return None, f"Failed to parse Parquet file: {str(e)}"
    else:
        # Reuse existing robust CSV parser from src.ingestion
        return load_csv(io.BytesIO(data_bytes))


def upload_to_s3(
    df: pd.DataFrame,
    bucket: str,
    key: str,
    format: str = "parquet",
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    region_name: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    mock_dir: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Serializes and uploads a DataFrame to an S3 bucket or local mock storage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to serialize and upload.
    bucket : str
        Target S3 bucket name.
    key : str
        Target S3 object key / destination path.
    format : str, default 'parquet'
        Output format: 'parquet' or 'csv'.
    aws_access_key_id : str, optional
        AWS Access Key ID override.
    aws_secret_access_key : str, optional
        AWS Secret Access Key override.
    region_name : str, optional
        AWS Region override.
    endpoint_url : str, optional
        Optional custom S3 endpoint URL.
    mock_dir : str, optional
        Optional custom local mock root directory.

    Returns
    -------
    Tuple[bool, str]
        (True, destination_uri) on success, or (False, error_message) on failure.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False, "Input must be a valid pandas DataFrame."
    if df.empty:
        return False, "Cannot upload an empty DataFrame."

    clean_bucket = bucket.replace("s3://", "").strip("/\\") if bucket else ""
    clean_key = key.lstrip("/\\") if key else ""

    if not clean_bucket or not clean_key:
        return False, "Both bucket name and object key must be provided."

    fmt = format.lower().strip()
    if fmt == "parquet":
        try:
            data_bytes = to_parquet_bytes(df)
            content_type = "application/octet-stream"
        except Exception as e:
            return False, f"Failed to serialize DataFrame to Parquet: {str(e)}"
    elif fmt == "csv":
        try:
            data_bytes = to_csv_bytes(df)
            content_type = "text/csv"
        except Exception as e:
            return False, f"Failed to serialize DataFrame to CSV: {str(e)}"
    else:
        return False, f"Unsupported format '{format}'. Supported formats are 'parquet' and 'csv'."

    if is_mock_mode():
        target_path = get_mock_s3_path(clean_bucket, clean_key, mock_dir=mock_dir)
        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "wb") as f:
                f.write(data_bytes)
            return True, f"mock://{clean_bucket}/{clean_key}"
        except Exception as e:
            return False, f"Failed to write local mock file: {str(e)}"
    else:
        try:
            s3_client = get_s3_client(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name,
                endpoint_url=endpoint_url,
            )
            s3_client.put_object(
                Bucket=clean_bucket,
                Key=clean_key,
                Body=data_bytes,
                ContentType=content_type,
            )
            return True, f"s3://{clean_bucket}/{clean_key}"
        except NoCredentialsError:
            # Fall back to local mock storage
            target_path = get_mock_s3_path(clean_bucket, clean_key, mock_dir=mock_dir)
            try:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "wb") as f:
                    f.write(data_bytes)
                return True, f"mock://{clean_bucket}/{clean_key} (fallback: credentials not set)"
            except Exception as e:
                return (
                    False,
                    "AWS credentials not configured and failed to write fallback mock file: " + str(e),
                )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "Unknown")
            msg = e.response.get("Error", {}).get("Message", str(e))
            return False, f"S3 Error [{code}]: {msg}"
        except EndpointConnectionError as e:
            return False, f"Could not connect to S3 endpoint: {str(e)}"
        except Exception as e:
            return False, f"Failed to upload to s3://{clean_bucket}/{clean_key}: {str(e)}"
