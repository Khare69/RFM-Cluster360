"""
Unit tests for CSV ingestion, encoding detection, and fuzzy column auto-mapping.
"""

import io
import pandas as pd
import pytest

from src.ingestion import auto_map_columns, load_csv, validate_mapping


def test_load_csv_from_string_io():
    csv_text = "CustomerID,InvoiceNo,InvoiceDate,Quantity,UnitPrice\n1001,536365,2023-01-01,2,10.5\n"
    df, err = load_csv(io.BytesIO(csv_text.encode("utf-8")))
    assert err is None
    assert df is not None
    assert len(df) == 1
    assert "CustomerID" in df.columns


def test_load_csv_empty_file():
    empty_bytes = io.BytesIO(b"")
    df, err = load_csv(empty_bytes)
    assert df is None
    assert "empty" in err.lower()


def test_load_csv_latin1_encoding():
    # Character that differs in latin-1 vs utf-8 (e.g. accented characters)
    csv_latin1 = "CustomerID,InvoiceNo,InvoiceDate,Quantity,UnitPrice,Description\n1001,536365,2023-01-01,2,10.5,Café\n".encode("latin-1")
    df, err = load_csv(io.BytesIO(csv_latin1))
    assert err is None
    assert df is not None
    assert len(df) == 1


def test_auto_map_columns_exact():
    df = pd.DataFrame(columns=["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"])
    mapping = auto_map_columns(df)
    assert mapping["customer_id"] == "CustomerID"
    assert mapping["invoice_no"] == "InvoiceNo"
    assert mapping["invoice_date"] == "InvoiceDate"
    assert mapping["quantity"] == "Quantity"
    assert mapping["unit_price"] == "UnitPrice"


def test_auto_map_columns_fuzzy_and_aliases():
    df = pd.DataFrame(columns=["cust_id", "order_id", "created_at", "qty", "item_price", "region"])
    mapping = auto_map_columns(df)
    assert mapping["customer_id"] == "cust_id"
    assert mapping["invoice_no"] == "order_id"
    assert mapping["invoice_date"] == "created_at"
    assert mapping["quantity"] == "qty"
    assert mapping["unit_price"] == "item_price"
    assert mapping["country"] == "region"


def test_validate_mapping():
    complete_map = {
        "customer_id": "cust_id",
        "invoice_no": "order_id",
        "invoice_date": "date",
        "quantity": "qty",
        "unit_price": "price",
    }
    is_valid, missing = validate_mapping(complete_map)
    assert is_valid is True
    assert len(missing) == 0

    incomplete_map = {
        "customer_id": "cust_id",
        "invoice_no": None,
        "invoice_date": "date",
        "quantity": None,
        "unit_price": "price",
    }
    is_valid, missing = validate_mapping(incomplete_map)
    assert is_valid is False
    assert "invoice_no" in missing
    assert "quantity" in missing
