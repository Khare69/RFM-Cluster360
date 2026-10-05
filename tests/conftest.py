"""
Pytest configuration and shared fixtures for unit testing.
"""

from datetime import datetime
import pandas as pd
import pytest


@pytest.fixture
def raw_test_transactions() -> pd.DataFrame:
    """
    Returns a sample raw transactions DataFrame containing intentional data anomalies:
    - 2 customers with valid purchases
    - 1 cancelled order ('C104')
    - 1 row with missing CustomerID
    - 1 row with negative quantity
    - 1 row with zero price
    - 1 exact duplicate row
    """
    data = [
        # Customer 1001: 2 orders, 3 line items ($20 + $5 + $60 = $85)
        {"InvoiceNo": "101", "InvoiceDate": "2023-01-10 10:00:00", "CustomerID": 1001, "Quantity": 2, "UnitPrice": 10.0, "Country": "UK"},
        {"InvoiceNo": "101", "InvoiceDate": "2023-01-10 10:00:00", "CustomerID": 1001, "Quantity": 1, "UnitPrice": 5.0, "Country": "UK"},
        {"InvoiceNo": "102", "InvoiceDate": "2023-03-15 14:30:00", "CustomerID": 1001, "Quantity": 3, "UnitPrice": 20.0, "Country": "UK"},

        # Customer 1002: 1 order, $100 total
        {"InvoiceNo": "103", "InvoiceDate": "2023-01-05 09:00:00", "CustomerID": 1002, "Quantity": 1, "UnitPrice": 100.0, "Country": "Germany"},

        # Anomalies to be filtered out:
        # Cancelled order
        {"InvoiceNo": "C104", "InvoiceDate": "2023-02-01 11:00:00", "CustomerID": 1001, "Quantity": -2, "UnitPrice": 10.0, "Country": "UK"},
        # Missing CustomerID
        {"InvoiceNo": "105", "InvoiceDate": "2023-02-05 12:00:00", "CustomerID": None, "Quantity": 1, "UnitPrice": 15.0, "Country": "UK"},
        # Negative quantity
        {"InvoiceNo": "106", "InvoiceDate": "2023-02-10 13:00:00", "CustomerID": 1002, "Quantity": -1, "UnitPrice": 50.0, "Country": "Germany"},
        # Zero price
        {"InvoiceNo": "107", "InvoiceDate": "2023-02-15 14:00:00", "CustomerID": 1001, "Quantity": 5, "UnitPrice": 0.0, "Country": "UK"},
        # Exact duplicate of line item 1
        {"InvoiceNo": "101", "InvoiceDate": "2023-01-10 10:00:00", "CustomerID": 1001, "Quantity": 2, "UnitPrice": 10.0, "Country": "UK"},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def clean_test_transactions() -> pd.DataFrame:
    """
    Returns pre-cleaned transactions for 3 customers with deterministic RFM values.
    """
    data = [
        # Customer A: Bought 2023-03-15 (recent), 2 orders, $85 total
        {"customer_id": "A", "invoice_no": "101", "invoice_date": "2023-01-10", "quantity": 2, "unit_price": 10.0, "country": "UK"},
        {"customer_id": "A", "invoice_no": "101", "invoice_date": "2023-01-10", "quantity": 1, "unit_price": 5.0, "country": "UK"},
        {"customer_id": "A", "invoice_no": "102", "invoice_date": "2023-03-15", "quantity": 3, "unit_price": 20.0, "country": "UK"},

        # Customer B: Bought 2023-01-05 (stale), 1 order, $100 total
        {"customer_id": "B", "invoice_no": "103", "invoice_date": "2023-01-05", "quantity": 1, "unit_price": 100.0, "country": "Germany"},

        # Customer C: Bought 2023-03-10 (recent), 3 orders, $300 total
        {"customer_id": "C", "invoice_no": "104", "invoice_date": "2023-02-01", "quantity": 5, "unit_price": 20.0, "country": "France"},
        {"customer_id": "C", "invoice_no": "105", "invoice_date": "2023-02-20", "quantity": 5, "unit_price": 20.0, "country": "France"},
        {"customer_id": "C", "invoice_no": "106", "invoice_date": "2023-03-10", "quantity": 5, "unit_price": 20.0, "country": "France"},
    ]
    df = pd.DataFrame(data)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


@pytest.fixture
def synthetic_rfm_base() -> pd.DataFrame:
    """
    Returns an RFM dataset with 30 customers spanning distinct behavioral profiles for testing clustering.
    """
    import numpy as np
    np.random.seed(42)

    rows = []
    # 10 High value Champions: Low Recency (1-10), High Freq (10-25), High Monetary (1000-5000)
    for i in range(10):
        rows.append({
            "customer_id": f"C_CHAMP_{i}",
            "recency": float(np.random.randint(1, 15)),
            "frequency": int(np.random.randint(10, 30)),
            "monetary": float(np.random.uniform(1500, 5000)),
        })

    # 10 New / Casual: Low Recency (1-20), Low Freq (1-2), Low Monetary (20-100)
    for i in range(10):
        rows.append({
            "customer_id": f"C_NEW_{i}",
            "recency": float(np.random.randint(1, 25)),
            "frequency": int(np.random.randint(1, 3)),
            "monetary": float(np.random.uniform(20, 150)),
        })

    # 10 Lost / Dormant: High Recency (150-300), Low Freq (1-2), Low Monetary (20-100)
    for i in range(10):
        rows.append({
            "customer_id": f"C_LOST_{i}",
            "recency": float(np.random.randint(150, 320)),
            "frequency": int(np.random.randint(1, 3)),
            "monetary": float(np.random.uniform(20, 100)),
        })

    return pd.DataFrame(rows)
