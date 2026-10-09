"""
Application configuration and constants for RFM Cluster360.
All magic numbers, defaults, and column aliases are centralized here.
"""

import os
from typing import Dict, List, Tuple
from dotenv import load_dotenv

load_dotenv()

# File Limits & Encodings
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
SUPPORTED_ENCODINGS = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]

# Canonical column names required for RFM computation
REQUIRED_COLUMNS = [
    "customer_id",
    "invoice_no",
    "invoice_date",
    "quantity",
    "unit_price",
]

# Fuzzy matching / alias mapping for standard transaction files
COLUMN_ALIASES: Dict[str, List[str]] = {
    "customer_id": [
        "customerid",
        "customer_id",
        "cust_id",
        "clientid",
        "client_id",
        "customer",
        "user_id",
        "userid",
        "account_id",
        "accountid",
    ],
    "invoice_no": [
        "invoiceno",
        "invoice_no",
        "invoicenumber",
        "invoice_num",
        "invoice",
        "order_id",
        "orderid",
        "orderno",
        "order_number",
        "transaction_id",
        "trans_id",
        "bill_no",
    ],
    "invoice_date": [
        "invoicedate",
        "invoice_date",
        "date",
        "order_date",
        "orderdate",
        "transaction_date",
        "trans_date",
        "timestamp",
        "created_at",
        "purchase_date",
    ],
    "quantity": [
        "quantity",
        "qty",
        "count",
        "item_count",
        "volume",
        "units",
        "pieces",
    ],
    "unit_price": [
        "unitprice",
        "unit_price",
        "price",
        "item_price",
        "unit_cost",
        "rate",
        "amount",
        "sales",
    ],
    # Optional columns for richer slicing/filtering
    "country": [
        "country",
        "nation",
        "region",
        "location",
        "state",
    ],
    "description": [
        "description",
        "product_name",
        "item_description",
        "item_name",
        "product",
    ],
}

# Machine Learning & Clustering Settings
DEFAULT_K_MIN = int(os.getenv("DEFAULT_K_MIN", "2"))
DEFAULT_K_MAX = int(os.getenv("DEFAULT_K_MAX", "8"))
MIN_CUSTOMERS_FOR_CLUSTERING = int(os.getenv("MIN_CUSTOMERS_FOR_CLUSTERING", "10"))
RANDOM_STATE = 42

# Accessible Color Palette (Colorblind-friendly / Distinct)
SEGMENT_COLORS = {
    "Champions": "#1f77b4",          # Royal Blue
    "Loyal Customers": "#2ca02c",    # Forest Green
    "Potential Loyalists": "#17becf",# Cyan/Teal
    "New Customers": "#ff7f0e",      # Amber/Orange
    "Promising": "#bcbd22",          # Olive/Yellow-Green
    "Needs Attention": "#9467bd",    # Purple
    "At Risk": "#e377c2",            # Rose / Pink
    "About to Sleep": "#8c564b",     # Brown
    "Lost / Dormant": "#d62728",     # Crimson Red
}

DEFAULT_CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# Cloud Storage & AWS S3 Settings
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "")
USE_LOCAL_MOCK = os.getenv("USE_LOCAL_MOCK", "false").lower() in ("true", "1", "yes")
MOCK_S3_DIR = os.getenv(
    "MOCK_S3_DIR",
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mock_s3")),
)

# MLOps Model Registry Settings
MODEL_REGISTRY_DIR = os.getenv(
    "MODEL_REGISTRY_DIR",
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "models")),
)

