# Task 4: Data Visualization
# Dataset: E-Commerce Orders Dataset

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and prepare dataset
df = pd.read_excel("week2.csv")
df['CouponCode'] = df['CouponCode'].fillna('No Coupon')
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['MonthYear'] = df['Date'].dt.to_period('M').astype(str)

colors = ['steelblue', 'darkorange', 'seagreen', 'tomato', 'mediumpurple']

# ---------------------------------------------------------
# Chart 1: Revenue by Product (Horizontal Bar)
# ---------------------------------------------------------

product_revenue = df.groupby('Product')['TotalPrice'].sum().sort_values()

plt.figure(figsize=(9, 5))
bars = plt.barh(product_revenue.index, product_revenue.values, color='steelblue', edgecolor='black')
plt.xlabel("Total Revenue ($)")
plt.title("Total Revenue by Product")
for bar in bars:
    plt.text(bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
             f"${bar.get_width():,.0f}", va='center', fontsize=9)
plt.tight_layout()
plt.savefig("chart1_revenue_by_product.png", dpi=150)
plt.close()
print("Saved: chart1_revenue_by_product.png")

# ---------------------------------------------------------
# Chart 2: Orders by Payment Method (Bar Chart)
# ---------------------------------------------------------

payment_counts = df['PaymentMethod'].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(payment_counts.index, payment_counts.values, color=colors, edgecolor='black')
plt.xlabel("Payment Method")
plt.ylabel("Number of Orders")
plt.title("Orders by Payment Method")
plt.tight_layout()
plt.savefig("chart2_payment_methods.png", dpi=150)
plt.close()
print("Saved: chart2_payment_methods.png")

# ---------------------------------------------------------
# Chart 3: Orders by Referral Source (Bar Chart)
# ---------------------------------------------------------

referral_counts = df['ReferralSource'].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(referral_counts.index, referral_counts.values, color='steelblue', edgecolor='black')
plt.xlabel("Referral Source")
plt.ylabel("Number of Orders")
plt.title("Orders by Referral Source")
plt.tight_layout()
plt.savefig("chart3_referral_sources.png", dpi=150)
plt.close()
print("Saved: chart3_referral_sources.png")

# ---------------------------------------------------------
# Chart 4: Order Status Distribution (Pie Chart)
# ---------------------------------------------------------

status_counts = df['OrderStatus'].value_counts()

plt.figure(figsize=(7, 5))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
plt.title("Order Status Distribution")
plt.tight_layout()
plt.savefig("chart4_order_status.png", dpi=150)
plt.close()
print("Saved: chart4_order_status.png")

# ---------------------------------------------------------
# Chart 5: Monthly Revenue Trend (Line Chart)
# ---------------------------------------------------------

monthly_revenue = df.groupby('MonthYear')['TotalPrice'].sum().reset_index()
monthly_revenue = monthly_revenue.sort_values('MonthYear')

plt.figure(figsize=(12, 5))
plt.plot(monthly_revenue['MonthYear'], monthly_revenue['TotalPrice'],
         marker='o', color='steelblue', linewidth=2)
plt.fill_between(monthly_revenue['MonthYear'], monthly_revenue['TotalPrice'],
                 alpha=0.15, color='steelblue')
plt.xlabel("Month")
plt.ylabel("Total Revenue ($)")
plt.title("Monthly Revenue Trend")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("chart5_monthly_revenue.png", dpi=150)
plt.close()
print("Saved: chart5_monthly_revenue.png")

# ---------------------------------------------------------
# Chart 6: TotalPrice Distribution (Histogram)
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.hist(df['TotalPrice'], bins=30, color='steelblue', edgecolor='black', alpha=0.8)
plt.xlabel("Total Price ($)")
plt.ylabel("Number of Orders")
plt.title("Distribution of Order Total Price")
plt.tight_layout()
plt.savefig("chart6_price_distribution.png", dpi=150)
plt.close()
print("Saved: chart6_price_distribution.png")

# ---------------------------------------------------------
# Chart 7: Coupon Usage - Average Order Value Comparison
# ---------------------------------------------------------

coupon_avg = df.groupby('CouponCode')['TotalPrice'].mean().sort_values(ascending=False)

plt.figure(figsize=(9, 5))
plt.bar(coupon_avg.index, coupon_avg.values, color='seagreen', edgecolor='black')
plt.xlabel("Coupon Code")
plt.ylabel("Average Order Value ($)")
plt.title("Average Order Value by Coupon Code")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("chart7_coupon_comparison.png", dpi=150)
plt.close()
print("Saved: chart7_coupon_comparison.png")

print("\nAll charts saved successfully.")
