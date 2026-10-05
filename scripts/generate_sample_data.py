"""
Script to generate realistic demo retail transaction data with built-in data quality noise
(guest checkouts, cancellations, returns) to test and showcase the full cleaning and segmentation pipeline.
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_sample_data(filename="data/sample_retail.csv"):
    random.seed(42)
    np.random.seed(42)

    products = [
        ("POSTAGE", 3.75),
        ("WHITE HANGING HEART T-LIGHT HOLDER", 2.95),
        ("REGENCY CAKESTAND 3 TIER", 12.75),
        ("PARTY BUNTING", 4.95),
        ("LUNCH BAG RED RETROSPOT", 1.65),
        ("SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
        ("JUMBO BAG RED RETROSPOT", 2.08),
        ("NATURAL SLATE CHALKBOARD LARGE", 4.95),
        ("HEART OF WICKER SMALL", 1.65),
        ("VINTAGE SNAP CARDS", 0.85),
        ("ASSORTED COLOUR BIRD ORNAMENT", 1.69),
        ("PACK OF 72 RETROSPOT CAKE CASES", 0.55),
        ("WOODEN PICTURE FRAME WHITE FINISH", 2.55),
        ("RED RETROSPOT CHARLOTTE BAG", 0.85),
        ("WOODEN STAR CHRISTMAS DECORATION", 0.85),
        ("DOORMAT AIRMAIL", 7.95),
        ("ALARM CLOCK BAKELIKE RED", 3.75),
        ("ALARM CLOCK BAKELIKE GREEN", 3.75),
        ("PAPER CHAIN KIT 50'S CHRISTMAS", 2.95),
        ("HAND WARMER OWL DESIGN", 2.10),
    ]

    countries = ["United Kingdom", "Germany", "France", "Spain", "Australia", "Netherlands"]
    country_weights = [0.82, 0.06, 0.05, 0.03, 0.02, 0.02]

    # Create 120 customer profiles
    # Types:
    # 1. Champions (15 customers): acquired Jan-Mar, buy every 2-4 weeks, high quantities
    # 2. Loyal (25 customers): acquired Jan-Jun, buy every 4-8 weeks, regular spend
    # 3. New (20 customers): acquired Nov-Dec, 1-2 orders
    # 4. At Risk / Lost (35 customers): acquired Jan-Apr, stopped buying by Jun-Jul
    # 5. Occasional / Promising (25 customers): acquired scattered, 2-4 orders
    customers = []
    
    # 1. Champions: 1001-1015
    for cid in range(1001, 1016):
        country = random.choices(countries, weights=country_weights)[0]
        customers.append({
            "id": cid,
            "type": "champion",
            "country": country,
            "first_month": random.randint(1, 3),
            "order_count": random.randint(8, 16),
            "items_per_order": (3, 8),
            "qty_range": (4, 24),
        })

    # 2. Loyal: 1016-1040
    for cid in range(1016, 1041):
        country = random.choices(countries, weights=country_weights)[0]
        customers.append({
            "id": cid,
            "type": "loyal",
            "country": country,
            "first_month": random.randint(1, 5),
            "order_count": random.randint(5, 9),
            "items_per_order": (2, 5),
            "qty_range": (2, 10),
        })

    # 3. New Customers: 1041-1060
    for cid in range(1041, 1061):
        country = random.choices(countries, weights=country_weights)[0]
        customers.append({
            "id": cid,
            "type": "new",
            "country": country,
            "first_month": random.randint(11, 12),
            "order_count": random.randint(1, 2),
            "items_per_order": (1, 4),
            "qty_range": (1, 6),
        })

    # 4. At Risk / Lost: 1061-1095
    for cid in range(1061, 1096):
        country = random.choices(countries, weights=country_weights)[0]
        customers.append({
            "id": cid,
            "type": "lost_or_at_risk",
            "country": country,
            "first_month": random.randint(1, 4),
            "order_count": random.randint(1, 4),
            "items_per_order": (1, 4),
            "qty_range": (1, 8),
            "last_active_month": random.randint(3, 6),
        })

    # 5. Promising / Moderate: 1096-1120
    for cid in range(1096, 1121):
        country = random.choices(countries, weights=country_weights)[0]
        customers.append({
            "id": cid,
            "type": "promising",
            "country": country,
            "first_month": random.randint(4, 9),
            "order_count": random.randint(2, 5),
            "items_per_order": (2, 5),
            "qty_range": (2, 10),
        })

    rows = []
    invoice_seq = 536000

    base_year = 2023

    for cust in customers:
        c_type = cust["type"]
        cid = cust["id"]
        country = cust["country"]
        first_m = cust["first_month"]
        num_orders = cust["order_count"]

        # Generate timestamps for orders
        first_date = datetime(base_year, first_m, random.randint(1, 25), random.randint(9, 17), random.randint(0, 59))
        
        if c_type == "lost_or_at_risk":
            last_m = cust["last_active_month"]
            if last_m <= first_m:
                last_m = first_m + 1
            max_days = (datetime(base_year, last_m, 28) - first_date).days
            order_dates = [first_date + timedelta(days=int(d)) for d in np.linspace(0, max(1, max_days), num_orders)]
        elif c_type == "new":
            order_dates = [first_date + timedelta(days=random.randint(0, 15)) for _ in range(num_orders)]
        else:
            # Active throughout the year up to Dec
            end_date = datetime(base_year, 12, random.randint(1, 15), random.randint(9, 18))
            total_days = max(1, (end_date - first_date).days)
            spacing = total_days / max(1, num_orders)
            order_dates = [first_date + timedelta(days=int(i * spacing + random.uniform(-3, 3))) for i in range(num_orders)]
            # ensure all within 2023
            order_dates = [min(d, datetime(base_year, 12, 18, 16)) for d in order_dates]

        for o_date in order_dates:
            invoice_seq += 1
            inv_no = str(invoice_seq)
            item_count = random.randint(cust["items_per_order"][0], cust["items_per_order"][1])
            selected_products = random.sample(products, min(item_count, len(products)))

            for prod_name, unit_price in selected_products:
                qty = random.randint(cust["qty_range"][0], cust["qty_range"][1])
                rows.append({
                    "InvoiceNo": inv_no,
                    "Description": prod_name,
                    "Quantity": qty,
                    "InvoiceDate": o_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "UnitPrice": unit_price,
                    "CustomerID": cid,
                    "Country": country,
                })

    # Add realistic data quality noise:
    # 1. 25 guest checkouts (missing CustomerID)
    for _ in range(25):
        invoice_seq += 1
        prod_name, unit_price = random.choice(products)
        rows.append({
            "InvoiceNo": str(invoice_seq),
            "Description": prod_name,
            "Quantity": random.randint(1, 3),
            "InvoiceDate": datetime(base_year, random.randint(1, 12), random.randint(1, 28), 12, 0).strftime("%Y-%m-%d %H:%M:%S"),
            "UnitPrice": unit_price,
            "CustomerID": np.nan,  # Missing ID
            "Country": "United Kingdom",
        })

    # 2. 8 cancelled orders (InvoiceNo starts with 'C')
    for _ in range(8):
        prod_name, unit_price = random.choice(products)
        rows.append({
            "InvoiceNo": f"C{invoice_seq + random.randint(10, 50)}",
            "Description": prod_name,
            "Quantity": -random.randint(1, 5),
            "InvoiceDate": datetime(base_year, random.randint(3, 11), random.randint(1, 28), 14, 30).strftime("%Y-%m-%d %H:%M:%S"),
            "UnitPrice": unit_price,
            "CustomerID": random.randint(1001, 1050),
            "Country": "United Kingdom",
        })

    # 3. 5 zero/negative price or quantity rows (free gifts, damaged)
    for _ in range(5):
        invoice_seq += 1
        rows.append({
            "InvoiceNo": str(invoice_seq),
            "Description": "Damaged / Free Sample",
            "Quantity": 0,
            "InvoiceDate": datetime(base_year, 6, 15, 10, 0).strftime("%Y-%m-%d %H:%M:%S"),
            "UnitPrice": 0.0,
            "CustomerID": 1020,
            "Country": "United Kingdom",
        })

    # 4. 4 duplicate rows
    duplicates = random.sample(rows[:100], 4)
    rows.extend(duplicates)

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Generated {len(df)} rows across {df['CustomerID'].nunique()} unique customers -> {filename}")

if __name__ == "__main__":
    generate_sample_data()
