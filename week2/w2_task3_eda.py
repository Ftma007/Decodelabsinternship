# Task 3: Exploratory Data Analysis (EDA)
# Dataset: E-Commerce Orders Dataset

import pandas as pd

# Load and prepare dataset
df = pd.read_excel("week2.csv")
df['CouponCode'] = df['CouponCode'].fillna('No Coupon')
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

numeric_cols = ['Quantity', 'UnitPrice', 'TotalPrice', 'ItemsInCart']

# ---------------------------------------------------------
# Step 1: Basic Statistics
# ---------------------------------------------------------

print("Basic Descriptive Statistics:")
print(df[numeric_cols].describe().round(2))

# ---------------------------------------------------------
# Step 2: Sales Analysis
# ---------------------------------------------------------

print("\nTotal Revenue: ${:,.2f}".format(df['TotalPrice'].sum()))
print("Average Order Value: ${:,.2f}".format(df['TotalPrice'].mean()))
print("Highest Order Value: ${:,.2f}".format(df['TotalPrice'].max()))
print("Lowest Order Value : ${:,.2f}".format(df['TotalPrice'].min()))

print("\nRevenue by Product:")
product_revenue = df.groupby('Product')['TotalPrice'].sum().sort_values(ascending=False)
print(product_revenue.round(2))

print("\nTotal Orders per Product:")
print(df['Product'].value_counts())

# ---------------------------------------------------------
# Step 3: Customer & Payment Analysis
# ---------------------------------------------------------

print("\nOrders by Payment Method:")
print(df['PaymentMethod'].value_counts())

print("\nRevenue by Payment Method:")
print(df.groupby('PaymentMethod')['TotalPrice'].sum().sort_values(ascending=False).round(2))

print("\nOrders by Referral Source:")
print(df['ReferralSource'].value_counts())

# ---------------------------------------------------------
# Step 4: Order Status Analysis
# ---------------------------------------------------------

print("\nOrder Status Distribution:")
print(df['OrderStatus'].value_counts())

print("\nRevenue by Order Status:")
print(df.groupby('OrderStatus')['TotalPrice'].sum().sort_values(ascending=False).round(2))

# ---------------------------------------------------------
# Step 5: Coupon Usage Analysis
# ---------------------------------------------------------

print("\nCoupon Code Usage:")
print(df['CouponCode'].value_counts())

coupon_used = df[df['CouponCode'] != 'No Coupon']
no_coupon = df[df['CouponCode'] == 'No Coupon']
print(f"\nAverage order value WITH coupon   : ${coupon_used['TotalPrice'].mean():.2f}")
print(f"Average order value WITHOUT coupon: ${no_coupon['TotalPrice'].mean():.2f}")

# ---------------------------------------------------------
# Step 6: Monthly Trends
# ---------------------------------------------------------

print("\nMonthly Order Count:")
print(df.groupby(['Year', 'Month'])['OrderID'].count().to_string())

print("\nMonthly Revenue:")
print(df.groupby(['Year', 'Month'])['TotalPrice'].sum().round(2).to_string())

# ---------------------------------------------------------
# Step 7: Outlier Detection
# ---------------------------------------------------------

print("\nOutlier Detection (IQR Method):")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    count = len(df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)])
    print(f"  {col}: {count} outlier(s)")

# ---------------------------------------------------------
# Step 8: Summary of Findings
# ---------------------------------------------------------

top_product = product_revenue.idxmax()
top_referral = df['ReferralSource'].value_counts().idxmax()
top_payment = df['PaymentMethod'].value_counts().idxmax()

print("\nSummary of Key Findings:")
print(f"""
  1. The dataset contains 1200 orders with a total revenue of
     ${df['TotalPrice'].sum():,.2f} and an average order value of
     ${df['TotalPrice'].mean():,.2f}.

  2. '{top_product}' is the top-selling product by total revenue.

  3. '{top_referral}' is the most common referral source,
     suggesting it is the most effective marketing channel.

  4. '{top_payment}' is the most used payment method.

  5. {len(coupon_used)} out of 1200 orders used a coupon code
     ({len(coupon_used)/1200*100:.1f}% of all orders).

  6. Customers who used a coupon had an average order value of
     ${coupon_used['TotalPrice'].mean():.2f} compared to
     ${no_coupon['TotalPrice'].mean():.2f} for those without a coupon.
""")
