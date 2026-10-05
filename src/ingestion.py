"""
Module for file ingestion, encoding fallback detection, and fuzzy column auto-mapping.
"""

import io
import difflib
import re
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from config.settings import (
    COLUMN_ALIASES,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    REQUIRED_COLUMNS,
    SUPPORTED_ENCODINGS,
)


def _normalize_name(name: str) -> str:
    """Normalize string by lowercasing and stripping non-alphanumeric chars."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_csv(
    file_source: Union[str, bytes, io.BytesIO, io.StringIO, object],
    max_size_mb: int = MAX_FILE_SIZE_MB,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load a CSV file with automatic encoding fallback and size validation.

    Parameters
    ----------
    file_source : str, bytes, BytesIO, or file-like object
        The path or file object to read.
    max_size_mb : int, default MAX_FILE_SIZE_MB
        Maximum allowed file size in MB.

    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[str]]
        (DataFrame, None) on success, or (None, error_message) on failure.
    """
    try:
        # Check size if available
        if hasattr(file_source, "size") and getattr(file_source, "size", 0) > (max_size_mb * 1024 * 1024):
            return None, f"File exceeds maximum allowed size of {max_size_mb} MB."

        # Get raw bytes or read directly
        if isinstance(file_source, str):
            # File path
            for encoding in SUPPORTED_ENCODINGS:
                try:
                    df = pd.read_csv(file_source, encoding=encoding, low_memory=False)
                    return df, None
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception as e:
                    return None, f"Failed to read CSV: {str(e)}"
            return None, f"Could not decode file using supported encodings: {', '.join(SUPPORTED_ENCODINGS)}"

        # File-like object or bytes
        if hasattr(file_source, "getvalue"):
            raw_bytes = file_source.getvalue()
        elif hasattr(file_source, "read"):
            raw_bytes = file_source.read()
            if hasattr(file_source, "seek"):
                file_source.seek(0)
        elif isinstance(file_source, (bytes, bytearray)):
            raw_bytes = bytes(file_source)
        else:
            return None, "Unsupported file source type."

        if len(raw_bytes) > (max_size_mb * 1024 * 1024):
            return None, f"File exceeds maximum allowed size of {max_size_mb} MB."

        if len(raw_bytes) == 0:
            return None, "The uploaded file is completely empty."

        for encoding in SUPPORTED_ENCODINGS:
            try:
                df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, low_memory=False)
                if df.empty:
                    return None, "The CSV file contains no data rows."
                return df, None
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                return None, f"Failed to parse CSV: {str(e)}"

        return None, f"Could not decode file using supported encodings: {', '.join(SUPPORTED_ENCODINGS)}"

    except Exception as e:
        return None, f"Unexpected error loading CSV: {str(e)}"


def auto_map_columns(
    df: pd.DataFrame,
    alias_dict: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Optional[str]]:
    """
    Automatically maps DataFrame column names to canonical schema names using aliases & fuzzy matching.

    Parameters
    ----------
    df : pd.DataFrame
        Uploaded DataFrame.
    alias_dict : dict, optional
        Canonical column -> list of alias variants. Defaults to COLUMN_ALIASES.

    Returns
    -------
    Dict[str, Optional[str]]
        Mapping of canonical_name -> actual_df_column (or None if unmapped).
    """
    aliases = alias_dict if alias_dict is not None else COLUMN_ALIASES
    actual_cols = list(df.columns)
    mapping: Dict[str, Optional[str]] = {key: None for key in aliases.keys()}
    assigned_actual_cols = set()

    # Pass 1: Exact normalized matching
    normalized_actual = {_normalize_name(col): col for col in actual_cols}

    for canonical, alias_list in aliases.items():
        # Check canonical name itself
        norm_canonical = _normalize_name(canonical)
        if norm_canonical in normalized_actual and normalized_actual[norm_canonical] not in assigned_actual_cols:
            actual_col = normalized_actual[norm_canonical]
            mapping[canonical] = actual_col
            assigned_actual_cols.add(actual_col)
            continue

        # Check aliases
        for alias in alias_list:
            norm_alias = _normalize_name(alias)
            if norm_alias in normalized_actual and normalized_actual[norm_alias] not in assigned_actual_cols:
                actual_col = normalized_actual[norm_alias]
                mapping[canonical] = actual_col
                assigned_actual_cols.add(actual_col)
                break

    # Pass 2: Fuzzy matching for remaining unmapped canonicals
    remaining_actual = [col for col in actual_cols if col not in assigned_actual_cols]
    if remaining_actual:
        normalized_remaining_map = {_normalize_name(col): col for col in remaining_actual}
        norm_remaining_keys = list(normalized_remaining_map.keys())

        for canonical, alias_list in aliases.items():
            if mapping[canonical] is not None:
                continue

            candidates = [_normalize_name(canonical)] + [_normalize_name(a) for a in alias_list]
            best_match_col = None

            for cand in candidates:
                close = difflib.get_close_matches(cand, norm_remaining_keys, n=1, cutoff=0.75)
                if close:
                    best_match_col = normalized_remaining_map[close[0]]
                    break

            if best_match_col and best_match_col not in assigned_actual_cols:
                mapping[canonical] = best_match_col
                assigned_actual_cols.add(best_match_col)

    return mapping


def validate_mapping(
    mapping: Dict[str, Optional[str]],
    required_cols: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validates whether all required canonical columns have been mapped to actual columns.

    Parameters
    ----------
    mapping : dict
        Canonical name -> actual column name.
    required_cols : list, optional
        List of required canonical names. Defaults to REQUIRED_COLUMNS.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list of missing column names)
    """
    reqs = required_cols if required_cols is not None else REQUIRED_COLUMNS
    missing = [col for col in reqs if not mapping.get(col)]
    return (len(missing) == 0, missing)
