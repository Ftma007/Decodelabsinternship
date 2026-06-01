# Task 1: Data Collection & Dataset Understanding

import pandas as pd

# Load the dataset
df = pd.read_excel("week2.csv")

# ---------------------------------------------------------
# Basic Dataset Info
# ---------------------------------------------------------

print("Dataset Shape:")
print(f"  Rows   : {df.shape[0]}")
print(f"  Columns: {df.shape[1]}")

print("\nColumn Names and Data Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

# ---------------------------------------------------------
# Understanding the Features
# ---------------------------------------------------------

print("\nColumn Descriptions:")
print("  OrderID        - Unique identifier for each order (string)")
print("  Date           - Date the order was placed (datetime)")
print("  CustomerID     - Unique identifier for each customer (string)")
print("  Product        - Name of the product ordered (string)")
print("  Quantity       - Number of units ordered (integer)")
print("  UnitPrice      - Price per unit in dollars (float)")
print("  ShippingAddress- City/region where the order was shipped (string)")
print("  PaymentMethod  - Method used to pay (string)")
print("  OrderStatus    - Current status of the order (string)")
print("  TrackingNumber - Shipment tracking number (string)")
print("  ItemsInCart    - Total items in the cart at time of order (integer)")
print("  CouponCode     - Discount coupon applied, if any (string)")
print("  ReferralSource - How the customer found the store (string)")
print("  TotalPrice     - Final total price of the order in dollars (float)")

# ---------------------------------------------------------
# Unique Values in Categorical Columns
# ---------------------------------------------------------

print("\nUnique Products:")
print(df['Product'].unique())

print("\nUnique Payment Methods:")
print(df['PaymentMethod'].unique())

print("\nUnique Order Statuses:")
print(df['OrderStatus'].unique())

print("\nUnique Referral Sources:")
print(df['ReferralSource'].unique())

print("\nUnique Coupon Codes:")
print(df['CouponCode'].unique())

print("\nDate Range of Orders:")
print(f"  From : {df['Date'].min().date()}")
print(f"  To   : {df['Date'].max().date()}")

# ---------------------------------------------------------
# Dataset Summary
# ---------------------------------------------------------

print("\nDataset Overview:")
print("""
  This dataset contains 1200 e-commerce orders with information
  about customers, products, payments, shipping, and order status.
  It covers orders placed between 2023 and 2024 and includes
  details useful for sales analysis, customer behavior study,
  and business performance tracking.
""")
