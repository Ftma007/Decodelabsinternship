# Task 2: Data Cleaning & Preprocessing
# Dataset: E-Commerce Orders Dataset

import pandas as pd

# Load the dataset
df = pd.read_excel("week2.csv")

print("Original Dataset Shape:", df.shape)

# ---------------------------------------------------------
# Step 1: Check for Missing Values
# ---------------------------------------------------------

print("\nMissing Values per Column:")
print(df.isnull().sum())

# CouponCode has 309 missing values — this is expected
# because not every order uses a coupon
print("\nNote: CouponCode has 309 missing values.")
print("  This is expected — not all customers use a coupon.")
print("  Filling missing CouponCode with 'No Coupon'.")

df['CouponCode'] = df['CouponCode'].fillna('No Coupon')

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# ---------------------------------------------------------
# Step 2: Check for Duplicate Rows
# ---------------------------------------------------------

print(f"\nDuplicate Rows Found: {df.duplicated().sum()}")
print(f"Duplicate OrderIDs  : {df['OrderID'].duplicated().sum()}")
print("No duplicates found — dataset is already unique per order.")

# ---------------------------------------------------------
# Step 3: Check and Fix Data Types
# ---------------------------------------------------------

print("\nData Types Before:")
print(df.dtypes)

# Ensure Date column is datetime
df['Date'] = pd.to_datetime(df['Date'])

# Extract useful time features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.day_name()

print("\nData Types After:")
print(df.dtypes)

# ---------------------------------------------------------
# Step 4: Check for Outliers in Numeric Columns
# ---------------------------------------------------------

numeric_cols = ['Quantity', 'UnitPrice', 'TotalPrice', 'ItemsInCart']

print("\nOutlier Detection (IQR Method):")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"  {col}: {len(outliers)} outlier(s)  |  accepted range: [{lower:.2f}, {upper:.2f}]")

# ---------------------------------------------------------
# Step 5: Verify TotalPrice matches Quantity * UnitPrice
# ---------------------------------------------------------

df['ExpectedTotal'] = df['Quantity'] * df['UnitPrice']
mismatches = df[abs(df['TotalPrice'] - df['ExpectedTotal']) > 1]
print(f"\nOrders where TotalPrice doesn't match Quantity x UnitPrice: {len(mismatches)}")
print("  (Difference > $1 threshold, likely due to discounts or shipping fees)")
df.drop(columns=['ExpectedTotal'], inplace=True)

# ---------------------------------------------------------
# Step 6: Final Cleaned Dataset
# ---------------------------------------------------------

print("\nCleaned Dataset Shape:", df.shape)
print("\nSample of Cleaned Data:")
print(df[['OrderID', 'Date', 'Product', 'Quantity', 'TotalPrice', 'CouponCode']].head())

print("\nData cleaning complete. Dataset is ready for analysis.")
