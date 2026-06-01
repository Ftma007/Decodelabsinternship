# Task 5: Predictive Model - Order Revenue Prediction (Linear Regression)
# Dataset: E-Commerce Orders Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load and prepare dataset
df = pd.read_excel("week2.csv")
df['CouponCode'] = df['CouponCode'].fillna('No Coupon')
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# ---------------------------------------------------------
# Step 1: Feature Engineering
# ---------------------------------------------------------

# Encode categorical columns
le_product = LabelEncoder()
le_payment = LabelEncoder()
le_coupon = LabelEncoder()
le_referral = LabelEncoder()

df['Product_enc'] = le_product.fit_transform(df['Product'])
df['Payment_enc'] = le_payment.fit_transform(df['PaymentMethod'])
df['Coupon_enc'] = le_coupon.fit_transform(df['CouponCode'])
df['Referral_enc'] = le_referral.fit_transform(df['ReferralSource'])

print("Encoded Labels:")
print("  Products :", list(le_product.classes_))
print("  Payments :", list(le_payment.classes_))
print("  Coupons  :", list(le_coupon.classes_))
print("  Referrals:", list(le_referral.classes_))

# ---------------------------------------------------------
# Step 2: Define Features and Target
# ---------------------------------------------------------

features = ['Quantity', 'UnitPrice', 'ItemsInCart', 'Month',
            'Product_enc', 'Payment_enc', 'Coupon_enc', 'Referral_enc']

X = df[features]
y = df['TotalPrice']

print(f"\nFeatures used: {features}")
print(f"Target variable: TotalPrice")

# ---------------------------------------------------------
# Step 3: Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# ---------------------------------------------------------
# Step 4: Train the Linear Regression Model
# ---------------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel: Linear Regression")
print("Training complete.")

# ---------------------------------------------------------
# Step 5: Evaluate the Model
# ---------------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"  Mean Absolute Error (MAE) : ${mae:.2f}")
print(f"  Root Mean Squared Error   : ${rmse:.2f}")
print(f"  R-squared Score           : {r2:.4f}")
print(f"\n  R2 of {r2:.2f} means the model explains {r2*100:.1f}% of the")
print(f"  variance in TotalPrice.")

# ---------------------------------------------------------
# Step 6: Feature Importance (Coefficients)
# ---------------------------------------------------------

print("\nFeature Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"  {feature:15s}: {coef:.4f}")
print(f"  Intercept      : {model.intercept_:.4f}")

# ---------------------------------------------------------
# Step 7: Actual vs Predicted Plot
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue', s=30)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linewidth=1.5, linestyle='--', label='Perfect Prediction')
plt.xlabel("Actual Total Price ($)")
plt.ylabel("Predicted Total Price ($)")
plt.title(f"Actual vs Predicted Order Value  (R2 = {r2:.2f})")
plt.legend()
plt.tight_layout()
plt.savefig("chart8_actual_vs_predicted.png", dpi=150)
plt.close()
print("\nSaved: chart8_actual_vs_predicted.png")

# ---------------------------------------------------------
# Step 8: Sample Prediction
# ---------------------------------------------------------

print("\nSample Prediction:")
sample = pd.DataFrame([{
    'Quantity': 3,
    'UnitPrice': 250.0,
    'ItemsInCart': 4,
    'Month': 6,
    'Product_enc': 0,
    'Payment_enc': 1,
    'Coupon_enc': 0,
    'Referral_enc': 2
}])

predicted_price = model.predict(sample)[0]
print(f"  Input  : Quantity=3, UnitPrice=$250, ItemsInCart=4, Month=June")
print(f"  Predicted Total Price: ${predicted_price:.2f}")
