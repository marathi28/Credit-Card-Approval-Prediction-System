"""
app.py
-------
Step 6 of the project: Building the Flask Web Application.

Routes:
    /            Home page introduction (Overview)
    /predict     Credit Card Approval Prediction Interface (GET form, POST result)
    /models      Model comparison ("Model Ledger")

Loads the best-performing model, scaler, and label encoders produced by
model/train_models.py and data/preprocess.py, and serves real-time
predictions through a simple, form-based UI.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

app = Flask(__name__)

# ---------------------------------------------------------------------
# Load trained artifacts once at startup
# ---------------------------------------------------------------------
best_model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

with open(os.path.join(MODEL_DIR, "best_model_name.txt")) as f:
    BEST_MODEL_NAME = f.read().strip()

with open(os.path.join(MODEL_DIR, "model_comparison.json")) as f:
    MODEL_COMPARISON = json.load(f)

processed_df = pd.read_csv(os.path.join(DATA_DIR, "processed_data.csv"))
N_RECORDS = len(processed_df)
BEST_AUC = round(MODEL_COMPARISON[BEST_MODEL_NAME]["roc_auc"], 3)

SKILLS = [
    "XGBoost", "Machine Learning Algorithms", "Artificial Intelligence",
    "Decision Tree Learning", "NumPy", "Python", "Scikit-Learn", "Matplotlib",
    "Flask", "Pandas", "Seaborn",
]

# Dropdown option lists pulled straight from the fitted label encoders so the
# UI always matches what the model was trained on.
INCOME_TYPES = list(label_encoders["NAME_INCOME_TYPE"].classes_)
EDUCATION_TYPES = list(label_encoders["NAME_EDUCATION_TYPE"].classes_)
FAMILY_STATUSES = list(label_encoders["NAME_FAMILY_STATUS"].classes_)
HOUSING_TYPES = list(label_encoders["NAME_HOUSING_TYPE"].classes_)
OCCUPATION_TYPES = list(label_encoders["OCCUPATION_TYPE"].classes_)


def encode_value(col, value):
    """Safely transform a raw categorical value using the fitted encoder,
    falling back to the first known class if an unseen value is submitted."""
    le = label_encoders[col]
    if value not in le.classes_:
        value = le.classes_[0]
    return int(le.transform([value])[0])


@app.route("/")
def home():
    return render_template(
        "index.html",
        active="home",
        n_records=f"{N_RECORDS:,}",
        best_auc=BEST_AUC,
        best_model_name=BEST_MODEL_NAME,
        skills=SKILLS,
    )


@app.route("/models")
def models_page():
    return render_template(
        "models.html",
        active="models",
        models=MODEL_COMPARISON,
        best_model_name=BEST_MODEL_NAME,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    if request.method == "GET":
        return render_template(
            "predict.html",
            active="predict",
            form={},
            income_types=INCOME_TYPES,
            education_types=EDUCATION_TYPES,
            family_statuses=FAMILY_STATUSES,
            housing_types=HOUSING_TYPES,
            occupation_types=OCCUPATION_TYPES,
            best_model_name=BEST_MODEL_NAME,
        )

    # ---------------- POST: build feature vector & predict ----------------
    f = request.form

    row = {
        "CODE_GENDER_ENC": encode_value("CODE_GENDER", f["gender"]),
        "FLAG_OWN_CAR_ENC": encode_value("FLAG_OWN_CAR", f["own_car"]),
        "FLAG_OWN_REALTY_ENC": encode_value("FLAG_OWN_REALTY", f["own_realty"]),
        "CNT_CHILDREN": int(f["children"]),
        "AMT_INCOME_TOTAL": float(f["income"]),
        "NAME_INCOME_TYPE_ENC": encode_value("NAME_INCOME_TYPE", f["income_type"]),
        "NAME_EDUCATION_TYPE_ENC": encode_value("NAME_EDUCATION_TYPE", f["education"]),
        "NAME_FAMILY_STATUS_ENC": encode_value("NAME_FAMILY_STATUS", f["family_status"]),
        "NAME_HOUSING_TYPE_ENC": encode_value("NAME_HOUSING_TYPE", f["housing_type"]),
        "AGE_YEARS": int(f["age"]),
        "EMPLOYMENT_YEARS": float(f["employment_years"]),
        "IS_CURRENTLY_EMPLOYED": int(f["is_employed"]),
        "FLAG_WORK_PHONE": int(f.get("work_phone", 0)),
        "FLAG_PHONE": int(f.get("phone", 0)),
        "FLAG_EMAIL": int(f.get("email", 0)),
        "OCCUPATION_TYPE_ENC": encode_value("OCCUPATION_TYPE", f["occupation"]),
        "CNT_FAM_MEMBERS": int(f["fam_members"]),
    }

    X = pd.DataFrame([row])[feature_columns]
    X_scaled = scaler.transform(X)

    prediction = int(best_model.predict(X_scaled)[0])       # 0 = approved, 1 = rejected
    risk_prob = float(best_model.predict_proba(X_scaled)[0, 1])
    risk_pct = round(risk_prob * 100, 1)

    # Lightweight, human-readable "what moved this decision" factors,
    # derived from simple domain heuristics rather than raw model internals
    # (keeps the explanation readable for a non-technical analyst).
    factors = []
    if float(f["income"]) < 150000:
        factors.append({"label": "Below-average annual income", "direction": "up", "arrow": "▲", "direction_label": "raises risk"})
    else:
        factors.append({"label": "Solid annual income", "direction": "down", "arrow": "▼", "direction_label": "lowers risk"})

    if f["own_realty"] == "Y":
        factors.append({"label": "Owns real estate", "direction": "down", "arrow": "▼", "direction_label": "lowers risk"})
    else:
        factors.append({"label": "No real estate owned", "direction": "up", "arrow": "▲", "direction_label": "raises risk"})

    if int(f["is_employed"]) == 0:
        factors.append({"label": "Not currently employed", "direction": "up", "arrow": "▲", "direction_label": "raises risk"})
    else:
        factors.append({"label": "Currently employed", "direction": "down", "arrow": "▼", "direction_label": "lowers risk"})

    if int(f["children"]) >= 3:
        factors.append({"label": "3+ dependents", "direction": "up", "arrow": "▲", "direction_label": "raises risk"})
    else:
        factors.append({"label": "Manageable dependents", "direction": "down", "arrow": "▼", "direction_label": "lowers risk"})

    snapshot = {
        "gender": "Male" if f["gender"] == "M" else "Female",
        "age": f["age"],
        "income": float(f["income"]),
        "income_type": f["income_type"],
        "education": f["education"],
        "employment_years": f["employment_years"],
        "family_status": f["family_status"],
        "housing_type": f["housing_type"],
        "own_realty": "Yes" if f["own_realty"] == "Y" else "No",
        "own_car": "Yes" if f["own_car"] == "Y" else "No",
    }

    return render_template(
        "result.html",
        active="predict",
        prediction=prediction,
        risk_pct=risk_pct,
        model_name=BEST_MODEL_NAME,
        snapshot=snapshot,
        factors=factors,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
