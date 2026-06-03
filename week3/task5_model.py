# Task 5: Predictive Model - Rain Classification (Random Forest)
# Decodelabs Data Science Internship - Week 3
# Dataset: Weather Forecast Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load and clean dataset
df = pd.read_csv("weather_forecast_data.csv")
df = df.drop_duplicates().reset_index(drop=True)

numeric_cols = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']

# ---------------------------------------------------------
# Step 1: Prepare Features and Target
# ---------------------------------------------------------

X = df[numeric_cols]

le = LabelEncoder()
y = le.fit_transform(df['Rain'])

print("Label Encoding:")
for i, label in enumerate(le.classes_):
    print(f"  {i} -> {label}")

# ---------------------------------------------------------
# Step 2: Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# ---------------------------------------------------------
# Step 3: Train Random Forest Classifier
# ---------------------------------------------------------

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nModel: Random Forest Classifier (100 trees)")
print("Training complete.")

# ---------------------------------------------------------
# Step 4: Evaluate the Model
# ---------------------------------------------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ---------------------------------------------------------
# Step 5: Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(le.classes_)
ax.set_yticklabels(le.classes_)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title(f"Confusion Matrix  (Accuracy: {accuracy:.0%})")
for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, cm[i, j], ha='center', va='center',
                fontsize=14, fontweight='bold', color=color)
plt.tight_layout()
plt.savefig("chart8_confusion_matrix.png", dpi=150)
plt.close()
print("\nSaved: chart8_confusion_matrix.png")

# ---------------------------------------------------------
# Step 6: Feature Importance
# ---------------------------------------------------------

importances = model.feature_importances_

print("\nFeature Importance:")
for feature, importance in sorted(zip(numeric_cols, importances),
                                   key=lambda x: x[1], reverse=True):
    print(f"  {feature:12s}: {importance:.4f}")

plt.figure(figsize=(7, 5))
sorted_idx = np.argsort(importances)
plt.barh([numeric_cols[i].replace('_', ' ') for i in sorted_idx],
         importances[sorted_idx], color='steelblue', edgecolor='black')
plt.xlabel("Importance Score")
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("chart9_feature_importance.png", dpi=150)
plt.close()
print("Saved: chart9_feature_importance.png")

# ---------------------------------------------------------
# Step 7: Sample Prediction
# ---------------------------------------------------------

print("\nSample Prediction:")
sample = pd.DataFrame([{
    'Temperature': 22.5,
    'Humidity': 85.0,
    'Wind_Speed': 6.0,
    'Cloud_Cover': 75.0,
    'Pressure': 995.0
}])
prediction = model.predict(sample)
probability = model.predict_proba(sample)[0]
print(f"  Input: Temp=22.5°C, Humidity=85%, Wind=6 km/h, Cloud=75%, Pressure=995 hPa")
print(f"  Predicted: {le.inverse_transform(prediction)[0]}")
print(f"  Confidence: no rain={probability[0]:.2%}, rain={probability[1]:.2%}")
