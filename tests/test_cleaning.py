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


def test_clean_transactions_case_insensitive_null_invoices():
    """Verify that case-insensitive placeholders ('None', 'NULL', '0') are caught and filtered."""
    data = [
        {"CustomerID": "1001", "InvoiceNo": "INV-001", "InvoiceDate": "2023-01-01", "Quantity": "1", "UnitPrice": "10.00"},
        {"CustomerID": "1001", "InvoiceNo": "None", "InvoiceDate": "2023-01-02", "Quantity": "2", "UnitPrice": "10.00"},
        {"CustomerID": "1002", "InvoiceNo": "NULL", "InvoiceDate": "2023-01-03", "Quantity": "1", "UnitPrice": "20.00"},
        {"CustomerID": "1002", "InvoiceNo": "0", "InvoiceDate": "2023-01-04", "Quantity": "1", "UnitPrice": "20.00"},
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
    assert clean_df.iloc[0]["invoice_no"] == "INV-001"


def test_clean_id_preserves_leading_zeros_and_distinct_formats():
    """Verifies leading zeros and non-decimal identifiers are strictly preserved, not falsely merged."""
    data = [
        {"CustomerID": "00123", "InvoiceNo": "101", "InvoiceDate": "2023-01-01", "Quantity": "1", "UnitPrice": "10.00"},
        {"CustomerID": "123", "InvoiceNo": "102", "InvoiceDate": "2023-01-02", "Quantity": "1", "UnitPrice": "10.00"},
        {"CustomerID": "00123.0", "InvoiceNo": "103", "InvoiceDate": "2023-01-03", "Quantity": "1", "UnitPrice": "10.00"},
        {"CustomerID": "1E5", "InvoiceNo": "104", "InvoiceDate": "2023-01-04", "Quantity": "1", "UnitPrice": "10.00"},
        {"CustomerID": "100000", "InvoiceNo": "105", "InvoiceDate": "2023-01-05", "Quantity": "1", "UnitPrice": "10.00"},
        {"CustomerID": "0", "InvoiceNo": "106", "InvoiceDate": "2023-01-06", "Quantity": "1", "UnitPrice": "10.00"},
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
    unique_ids = set(clean_df["customer_id"].unique())

    # "00123" and "00123.0" normalize to "00123"
    # "123" remains distinct from "00123"
    # "1E5" remains distinct from "100000"
    # "0" is not dropped as missing
    assert "00123" in unique_ids
    assert "123" in unique_ids
    assert "1E5" in unique_ids
    assert "100000" in unique_ids
    assert "0" in unique_ids
    assert report.total_customers == 5  # "00123", "123", "1E5", "100000", "0"
    assert report.rows_removed_missing_customer == 0


def test_clean_transactions_net_returns_policy():
    """Verify net returns policy retains return rows, tracks refunds, and calculates net revenue correctly."""
    from src.cleaning import CleaningPolicy

    data = {
        "InvoiceNo": ["1001", "C1002", "1003"],
        "CustomerID": ["C1", "C1", "C2"],
        "InvoiceDate": ["2023-01-01", "2023-01-05", "2023-01-02"],
        "Quantity": [2, -1, 5],
        "UnitPrice": [10.0, 10.0, 20.0],
    }
    df = pd.DataFrame(data)
    mapping = {
        "customer_id": "CustomerID",
        "invoice_no": "InvoiceNo",
        "invoice_date": "InvoiceDate",
        "quantity": "Quantity",
        "unit_price": "UnitPrice",
    }
    policy = CleaningPolicy(policy_type="net", return_time_window_days=30)
    clean_df, report = clean_transactions(df, mapping, policy=policy)

    assert report.policy_type == "net"
    assert report.refund_count == 1
    assert report.refund_amount == 10.0
    assert report.gross_revenue == 120.0  # 2*10 + 5*20
    assert report.total_revenue == 110.0  # 120 - 10
    assert round(report.refund_rate_pct, 2) == 8.33  # 10 / 120 * 100
    assert len(clean_df) == 3
    assert "is_refund" in clean_df.columns
    assert clean_df.loc[clean_df["invoice_no"] == "C1002", "is_refund"].iloc[0] == True


def test_export_dataframe_with_custom_currency():
    """Verify prepare_export_dataframe handles customizable currency symbols in column headers."""
    from src.export import prepare_export_dataframe

    rfm_df = pd.DataFrame([
        {
            "customer_id": "C1",
            "recency": 10,
            "frequency": 2,
            "monetary": 150.0,
            "r_score": 4,
            "f_score": 3,
            "m_score": 4,
            "rfm_segment": "434",
            "rfm_score": 3.7,
            "persona": "Champions",
            "cluster": 0,
        }
    ])

    export_eur = prepare_export_dataframe(rfm_df, currency_symbol="€")
    assert "Monetary (€)" in export_eur.columns
    assert "Monetary ($)" not in export_eur.columns

    export_inr = prepare_export_dataframe(rfm_df, currency_symbol="₹")
    assert "Monetary (₹)" in export_inr.columns
    assert "Monetary ($)" not in export_inr.columns




