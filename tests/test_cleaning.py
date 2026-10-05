"""
Unit tests for data validation, cleaning rules, and transparent metric reporting.
"""

import pandas as pd
import pytest

from src.cleaning import clean_transactions, detect_duplicates


def test_clean_transactions(raw_test_transactions):
    mapping = {
        "customer_id": "CustomerID",
        "invoice_no": "InvoiceNo",
        "invoice_date": "InvoiceDate",
        "quantity": "Quantity",
        "unit_price": "UnitPrice",
        "country": "Country",
    }

    clean_df, report = clean_transactions(raw_test_transactions, mapping, remove_duplicates=False)

    # Validate output shape and rows
    # Total input rows = 9
    # Removed:
    # 1 cancelled (C104)
    # 1 missing customer (105)
    # 1 negative qty (106)
    # 1 zero price (107)
    # Remaining = 5 rows (including the 1 duplicate)
    assert report.total_rows_before == 9
    assert report.total_rows_after == 5
    assert report.rows_removed_cancelled == 1
    assert report.rows_removed_missing_customer == 1
    assert report.rows_removed_non_positive_qty == 1
    assert report.rows_removed_non_positive_price == 1
    assert report.total_customers == 2  # 1001 and 1002

    # Revenue calculation:
    # 1001: (2*10) + (1*5) + (3*20) + (2*10 dup) = 20 + 5 + 60 + 20 = $105
    # 1002: (1*100) = $100
    # Total = $205
    assert report.total_revenue == 205.0

    # Ensure standardized column names
    for col in ["customer_id", "invoice_no", "invoice_date", "quantity", "unit_price", "country"]:
        assert col in clean_df.columns


def test_clean_transactions_remove_duplicates(raw_test_transactions):
    mapping = {
        "customer_id": "CustomerID",
        "invoice_no": "InvoiceNo",
        "invoice_date": "InvoiceDate",
        "quantity": "Quantity",
        "unit_price": "UnitPrice",
        "country": "Country",
    }

    clean_df, report = clean_transactions(raw_test_transactions, mapping, remove_duplicates=True)
    # Dropping the duplicate removes 1 row -> 4 rows remaining
    assert report.total_rows_after == 4
    # Revenue without duplicate: $85 (1001) + $100 (1002) = $185.0
    assert report.total_revenue == 185.0


def test_detect_duplicates(raw_test_transactions):
    dups = detect_duplicates(raw_test_transactions)
    assert len(dups) == 2  # The original and the duplicate copy


def test_clean_transactions_with_currency_strings():
    data = [
        {"CustomerID": "1001", "InvoiceNo": "101", "InvoiceDate": "2023-01-01", "Quantity": " 2 ", "UnitPrice": "$10.50"},
        {"CustomerID": "1002", "InvoiceNo": "102", "InvoiceDate": "2023-01-02", "Quantity": "1,000", "UnitPrice": "£2.50"},
    ]
    df = pd.DataFrame(data)
    mapping = {
        "customer_id": "CustomerID",
        "invoice_no": "InvoiceNo",
        "invoice_date": "InvoiceDate",
        "quantity": "Quantity",
        "unit_price": "UnitPrice",
    }
    clean_df, report = clean_transactions(df, mapping)
    assert report.total_rows_after == 2
    assert clean_df.iloc[0]["quantity"] == 2.0
    assert clean_df.iloc[0]["unit_price"] == 10.50
    assert clean_df.iloc[1]["quantity"] == 1000.0
    assert clean_df.iloc[1]["unit_price"] == 2.50
    assert report.total_revenue == (2 * 10.50) + (1000 * 2.50)


def test_clean_transactions_with_accounting_negatives():
    data = [
        {"CustomerID": "1001", "InvoiceNo": "101", "InvoiceDate": "2023-01-01", "Quantity": "5", "UnitPrice": "10.00"},
        {"CustomerID": "1001", "InvoiceNo": "102", "InvoiceDate": "2023-01-02", "Quantity": "(2)", "UnitPrice": "10.00"},
        {"CustomerID": "1002", "InvoiceNo": "103", "InvoiceDate": "2023-01-03", "Quantity": "1", "UnitPrice": "($15.00)"},
    ]
    df = pd.DataFrame(data)
    mapping = {
        "customer_id": "CustomerID",
        "invoice_no": "InvoiceNo",
        "invoice_date": "InvoiceDate",
        "quantity": "Quantity",
        "unit_price": "UnitPrice",
    }
    clean_df, report = clean_transactions(df, mapping)
    # The (2) quantity becomes -2 (filtered as non-positive qty)
    # The ($15.00) price becomes -15.00 (filtered as non-positive price)
    assert report.total_rows_after == 1
    assert report.rows_removed_non_positive_qty == 1
    assert report.rows_removed_non_positive_price == 1
    assert clean_df.iloc[0]["customer_id"] == "1001"
    assert clean_df.iloc[0]["quantity"] == 5.0


def test_clean_id_normalizes_float_strings():
    data = [
        {"CustomerID": 1001.0, "InvoiceNo": "101", "InvoiceDate": "2023-01-01", "Quantity": "1", "UnitPrice": "10.00"},
        {"CustomerID": "1001.0", "InvoiceNo": "102", "InvoiceDate": "2023-01-02", "Quantity": "2", "UnitPrice": "10.00"},
        {"CustomerID": "CUST-999", "InvoiceNo": "103", "InvoiceDate": "2023-01-03", "Quantity": "1", "UnitPrice": "20.00"},
    ]
    df = pd.DataFrame(data)
    mapping = {
        "customer_id": "CustomerID",
        "invoice_no": "InvoiceNo",
        "invoice_date": "InvoiceDate",
        "quantity": "Quantity",
        "unit_price": "UnitPrice",
    }
    clean_df, report = clean_transactions(df, mapping)
    assert report.total_customers == 2  # Customer "1001" and "CUST-999"
    assert set(clean_df["customer_id"].unique()) == {"1001", "CUST-999"}


def test_clean_transactions_null_invoice_no():
    data = [
        {"CustomerID": "1001", "InvoiceNo": "101", "InvoiceDate": "2023-01-01", "Quantity": "1", "UnitPrice": "10.00"},
        {"CustomerID": "1001", "InvoiceNo": None, "InvoiceDate": "2023-01-02", "Quantity": "2", "UnitPrice": "10.00"},
        {"CustomerID": "1002", "InvoiceNo": "   ", "InvoiceDate": "2023-01-03", "Quantity": "1", "UnitPrice": "20.00"},
        {"CustomerID": "1002", "InvoiceNo": "nan", "InvoiceDate": "2023-01-04", "Quantity": "1", "UnitPrice": "20.00"},
    ]
    df = pd.DataFrame(data)
    mapping = {
        "customer_id": "CustomerID",
        "invoice_no": "InvoiceNo",
        "invoice_date": "InvoiceDate",
        "quantity": "Quantity",
        "unit_price": "UnitPrice",
    }
    clean_df, report = clean_transactions(df, mapping)
    assert report.rows_removed_missing_invoice == 3
    assert report.total_rows_after == 1
    assert clean_df.iloc[0]["invoice_no"] == "101"


