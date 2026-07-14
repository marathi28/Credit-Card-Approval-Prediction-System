"""
preprocess.py
--------------
Step 4 of the project: Data Preprocessing & Feature Engineering.

- Loads application_record.csv and credit_record.csv
- Handles missing values
- Removes duplicate records
- Converts multi-class payment STATUS codes into a binary TARGET
  (0 = Approved / Good payer, 1 = Rejected / High-risk payer)
- Engineers AGE_YEARS and EMPLOYMENT_YEARS from DAYS_BIRTH / DAYS_EMPLOYED
- Encodes categorical variables into numerical format
- Outputs processed_data.csv ready for model training
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

app_df = pd.read_csv(os.path.join(DATA_DIR, "application_record.csv"))
credit_df = pd.read_csv(os.path.join(DATA_DIR, "credit_record.csv"))

print("Raw application shape:", app_df.shape)
print("Raw credit shape:", credit_df.shape)

# ---------------------------------------------------------------------
# 1. Convert multi-class payment STATUS into a binary risk label per ID
#    STATUS meaning (as in the public dataset):
#      0        : 1-29 days past due
#      1        : 30-59 days past due
#      2        : 60-89 days past due
#      3        : 90-119 days past due
#      4        : 120-149 days past due
#      5        : Overdue > 150 days (written off / bad debt)
#      C        : paid off that month
#      X        : no loan that month
#    An applicant is labeled HIGH-RISK (1) if they were ever 60+ days
#    past due (STATUS in 2,3,4,5). Otherwise they are GOOD (0).
# ---------------------------------------------------------------------
credit_df["IS_BAD_MONTH"] = credit_df["STATUS"].isin(["2", "3", "4", "5"]).astype(int)
bad_month_counts = credit_df.groupby("ID")["IS_BAD_MONTH"].sum().reset_index()
bad_month_counts["TARGET"] = (bad_month_counts["IS_BAD_MONTH"] >= 1).astype(int)
risk_per_id = bad_month_counts[["ID", "TARGET"]]
# TARGET = 1 -> high risk -> REJECTED ; TARGET = 0 -> APPROVED

df = app_df.merge(risk_per_id, on="ID", how="inner")
print("Merged shape:", df.shape)

# ---------------------------------------------------------------------
# 2. Handle missing values
# ---------------------------------------------------------------------
df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown")
df = df.dropna()

# ---------------------------------------------------------------------
# 3. Remove duplicate records
# ---------------------------------------------------------------------
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows")

# ---------------------------------------------------------------------
# 4. Feature engineering: convert day-based fields into human-readable
#    numeric features
# ---------------------------------------------------------------------
df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).astype(int)
df["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED"].apply(lambda x: 0 if x > 0 else round(-x / 365, 1))
df["IS_CURRENTLY_EMPLOYED"] = (df["DAYS_EMPLOYED"] < 0).astype(int)

# ---------------------------------------------------------------------
# 5. Encode categorical variables
# ---------------------------------------------------------------------
categorical_cols = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_ENC"] = le.fit_transform(df[col])
    encoders[col] = le

feature_cols = [
    "CODE_GENDER_ENC", "FLAG_OWN_CAR_ENC", "FLAG_OWN_REALTY_ENC", "CNT_CHILDREN",
    "AMT_INCOME_TOTAL", "NAME_INCOME_TYPE_ENC", "NAME_EDUCATION_TYPE_ENC",
    "NAME_FAMILY_STATUS_ENC", "NAME_HOUSING_TYPE_ENC", "AGE_YEARS",
    "EMPLOYMENT_YEARS", "IS_CURRENTLY_EMPLOYED", "FLAG_WORK_PHONE", "FLAG_PHONE",
    "FLAG_EMAIL", "OCCUPATION_TYPE_ENC", "CNT_FAM_MEMBERS",
]

final_df = df[feature_cols + ["TARGET"]].copy()
final_df.to_csv(os.path.join(DATA_DIR, "processed_data.csv"), index=False)

joblib.dump(encoders, os.path.join(DATA_DIR, "..", "model", "label_encoders.pkl"))
joblib.dump(feature_cols, os.path.join(DATA_DIR, "..", "model", "feature_columns.pkl"))

print("\nFinal processed shape:", final_df.shape)
print("Target distribution:\n", final_df["TARGET"].value_counts(normalize=True))
print("\nSaved: processed_data.csv, label_encoders.pkl, feature_columns.pkl")
