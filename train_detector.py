import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
data = pd.read_csv("data/transactions.csv")

# Features used by the detector
features = [
    "amount",
    "failed",
    "new_device",
    "international",
    "rapid_frequency",
]

X = data[features]
y = data["is_fraud"]

# Held-out test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Train an explainable Random Forest classifier
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)

# Predictions on held-out test set
y_pred = model.predict(X_test)

# Evaluation
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n===== FraudPulse AI Evaluation =====")
print(f"Training transactions: {len(X_train)}")
print(f"Held-out test transactions: {len(X_test)}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)

# Save model
joblib.dump(model, "model/fraud_detector.joblib")

print("\nModel saved to: model/fraud_detector.joblib")