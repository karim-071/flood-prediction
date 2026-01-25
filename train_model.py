import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from utils.data_loader import load_rainfall_data
from utils.features import add_rainfall_features, add_statistical_features
from utils.labeling import add_flood_label

# Load & prepare data
df = load_rainfall_data("data/rainfall.csv")
df = add_rainfall_features(df)
df = add_statistical_features(df)
df = add_flood_label(df)

# Select features
feature_cols = [
    "annual_rainfall",
    "monsoon_rainfall",
    "pre_monsoon_rainfall",
    "post_monsoon_rainfall",
    "monsoon_anomaly",
    "monsoon_percentile"
]

X = df[feature_cols]
y = df["flood_risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/flood_risk_model.pkl")
print("Model saved as flood_risk_model.pkl")
