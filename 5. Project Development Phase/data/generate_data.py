"""
generate_data.py
-----------------
Generates a realistic synthetic Credit Card Approval dataset that mirrors
the structure of the well-known public "Credit Card Approval Prediction"
dataset (application_record.csv + credit_record.csv), matching every
feature mentioned in the project brief:
    Gender, Income Type, Annual Income, Employment Duration, Education Level,
    Family Status, Housing Type, Age, Children, Family Members, and a
    multi-class monthly Payment Status history that gets converted into a
    binary approval label during feature engineering.

Run:
    python generate_data.py
Outputs:
    application_record.csv
    credit_record.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_APPLICANTS = 6000

genders = np.random.choice(["M", "F"], size=N_APPLICANTS, p=[0.45, 0.55])
own_car = np.random.choice(["Y", "N"], size=N_APPLICANTS, p=[0.4, 0.6])
own_realty = np.random.choice(["Y", "N"], size=N_APPLICANTS, p=[0.65, 0.35])
children = np.random.choice([0, 1, 2, 3, 4], size=N_APPLICANTS, p=[0.55, 0.22, 0.15, 0.06, 0.02])

income_types = np.random.choice(
    ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
    size=N_APPLICANTS, p=[0.52, 0.23, 0.14, 0.10, 0.01]
)

# Annual income correlated loosely with income type
base_income = {
    "Working": 180000, "Commercial associate": 240000, "Pensioner": 120000,
    "State servant": 160000, "Student": 40000
}
annual_income = np.array([
    max(30000, np.random.normal(base_income[t], base_income[t] * 0.35))
    for t in income_types
]).round(-2)

education = np.random.choice(
    ["Secondary / secondary special", "Higher education", "Incomplete higher",
     "Lower secondary", "Academic degree"],
    size=N_APPLICANTS, p=[0.45, 0.35, 0.10, 0.06, 0.04]
)

family_status = np.random.choice(
    ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
    size=N_APPLICANTS, p=[0.58, 0.20, 0.08, 0.08, 0.06]
)

housing_type = np.random.choice(
    ["House / apartment", "With parents", "Municipal apartment",
     "Rented apartment", "Office apartment", "Co-op apartment"],
    size=N_APPLICANTS, p=[0.70, 0.10, 0.08, 0.07, 0.03, 0.02]
)

age_days = -1 * (np.random.randint(21, 70, size=N_APPLICANTS) * 365 + np.random.randint(0, 365, size=N_APPLICANTS))

# Employment duration in days (negative = currently employed, large positive = pensioner/unemployed placeholder 365243)
employed_days = []
for t in income_types:
    if t == "Pensioner":
        employed_days.append(365243)  # sentinel used in the real dataset for "not employed"
    elif t == "Student":
        employed_days.append(-np.random.randint(1, 400))
    else:
        employed_days.append(-np.random.randint(30, 365 * 25))
employed_days = np.array(employed_days)

fam_members = children + np.where(np.isin(family_status, ["Married", "Civil marriage"]), 2, 1)

occupation_types = np.random.choice(
    ["Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
     "High skill tech staff", "Accountants", "Medicine staff", "Other"],
    size=N_APPLICANTS
)

flag_work_phone = np.random.choice([0, 1], size=N_APPLICANTS, p=[0.7, 0.3])
flag_phone = np.random.choice([0, 1], size=N_APPLICANTS, p=[0.6, 0.4])
flag_email = np.random.choice([0, 1], size=N_APPLICANTS, p=[0.8, 0.2])

applicant_ids = np.arange(1000001, 1000001 + N_APPLICANTS)

application_df = pd.DataFrame({
    "ID": applicant_ids,
    "CODE_GENDER": genders,
    "FLAG_OWN_CAR": own_car,
    "FLAG_OWN_REALTY": own_realty,
    "CNT_CHILDREN": children,
    "AMT_INCOME_TOTAL": annual_income,
    "NAME_INCOME_TYPE": income_types,
    "NAME_EDUCATION_TYPE": education,
    "NAME_FAMILY_STATUS": family_status,
    "NAME_HOUSING_TYPE": housing_type,
    "DAYS_BIRTH": age_days,
    "DAYS_EMPLOYED": employed_days,
    "FLAG_WORK_PHONE": flag_work_phone,
    "FLAG_PHONE": flag_phone,
    "FLAG_EMAIL": flag_email,
    "OCCUPATION_TYPE": occupation_types,
    "CNT_FAM_MEMBERS": fam_members,
})

# ---------------------------------------------------------------------------
# credit_record.csv : monthly payment history per applicant (multi-class
# STATUS code, as in the real dataset). We bias "risk" using income,
# employment duration and existing debt-like signals so the resulting
# binary label correlates sensibly with financial profile.
# ---------------------------------------------------------------------------
records = []

# Per-applicant probability of being a genuinely high-risk borrower,
# driven by income, employment stability, dependents and asset ownership.
employment_years_tmp = np.array([0 if d > 0 else -d / 365 for d in employed_days])
risk_score = (
    -0.7
    - 0.000012 * (annual_income - 180000)              # lower income -> higher risk
    + 1.4 * (employed_days > 0).astype(int)             # pensioners / unstable employment -> riskier
    - 0.05 * employment_years_tmp                       # longer tenure -> lower risk
    + 1.1 * (children >= 3).astype(int)
    - 0.9 * (own_realty == "Y").astype(int)
    - 0.5 * (own_car == "Y").astype(int)
    + np.random.normal(0, 0.35, size=N_APPLICANTS)
)
risk_prob = 1 / (1 + np.exp(-risk_score))
is_high_risk = np.random.binomial(1, risk_prob)  # ~15-20% of applicants

for idx, appl_id in enumerate(applicant_ids):
    months = np.random.randint(3, 25)
    bad = is_high_risk[idx]
    for m in range(months):
        if np.random.rand() < 0.35:
            status = "X"  # no active loan that month
        elif bad and np.random.rand() < 0.30:
            status = np.random.choice(["2", "3", "4", "5"], p=[0.5, 0.25, 0.15, 0.10])
        elif np.random.rand() < 0.15:
            status = "1"  # minor delinquency, not serious risk
        else:
            status = np.random.choice(["0", "C"], p=[0.3, 0.7])
        records.append((appl_id, -m, status))

credit_df = pd.DataFrame(records, columns=["ID", "MONTHS_BALANCE", "STATUS"])
print(f"High-risk applicant rate: {is_high_risk.mean():.2%}")

application_df.to_csv("application_record.csv", index=False)
credit_df.to_csv("credit_record.csv", index=False)

print(f"application_record.csv -> {application_df.shape}")
print(f"credit_record.csv      -> {credit_df.shape}")
print("Done.")
