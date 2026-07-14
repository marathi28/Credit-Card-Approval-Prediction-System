"""
train_models.py
-----------------
Step 5 of the project: Machine Learning Model Building.

Trains 4 classification algorithms:
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
    - XGBoost Classifier

Evaluates each with accuracy, confusion matrix, and classification report,
then saves the best-performing model (by ROC-AUC, more meaningful than
accuracy on this imbalanced task) plus the fitted scaler for deployment.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score
)
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "..", "data", "processed_data.csv")

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["TARGET"])
y = df["TARGET"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss", random_state=42, n_jobs=-1
    ),
}

results = {}
fitted_models = {}

print("=" * 70)
for name, model in models.items():
    # Logistic Regression benefits from scaled features; tree ensembles don't need it
    # but scaling doesn't hurt them, so we use scaled features consistently for
    # simplicity in the deployed pipeline.
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    results[name] = {"accuracy": acc, "roc_auc": auc, "confusion_matrix": cm.tolist(), "report": report}
    fitted_models[name] = model

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(classification_report(y_test, preds, target_names=["Approved", "Rejected"]))

# Select best model by ROC-AUC (robust to class imbalance)
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_model = fitted_models[best_name]

print("=" * 70)
print(f"BEST MODEL: {best_name} (ROC-AUC = {results[best_name]['roc_auc']:.4f})")
print("=" * 70)

# Save artifacts needed by the Flask app
joblib.dump(best_model, os.path.join(BASE, "best_model.pkl"))
joblib.dump(scaler, os.path.join(BASE, "scaler.pkl"))
with open(os.path.join(BASE, "best_model_name.txt"), "w") as f:
    f.write(best_name)

# Save a comparison summary (accuracy/auc only, JSON-safe) for the README/report
summary = {
    name: {"accuracy": round(r["accuracy"], 4), "roc_auc": round(r["roc_auc"], 4)}
    for name, r in results.items()
}
with open(os.path.join(BASE, "model_comparison.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved: best_model.pkl, scaler.pkl, best_model_name.txt, model_comparison.json")
