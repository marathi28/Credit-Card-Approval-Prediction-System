"""
eda.py
-------
Step 3 of the project: Data Visualization & Analysis.
Generates count plots and distribution plots and saves them to
static/images/ so they can also be shown in the Flask app if desired.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "..", "static", "images")
os.makedirs(OUT, exist_ok=True)

app_df = pd.read_csv(os.path.join(BASE, "application_record.csv"))
proc_df = pd.read_csv(os.path.join(BASE, "processed_data.csv"))

# 1. Target distribution
plt.figure(figsize=(5, 4))
sns.countplot(x="TARGET", data=proc_df, palette=["#2E7D32", "#C62828"])
plt.title("Approval (0) vs Rejection (1) Counts")
plt.xlabel("Target (0=Approved, 1=Rejected)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "target_distribution.png"), dpi=110)
plt.close()

# 2. Gender count plot
plt.figure(figsize=(5, 4))
sns.countplot(x="CODE_GENDER", data=app_df, palette="Set2")
plt.title("Applicant Gender Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "gender_count.png"), dpi=110)
plt.close()

# 3. Income type count plot
plt.figure(figsize=(7, 4))
sns.countplot(y="NAME_INCOME_TYPE", data=app_df,
              order=app_df["NAME_INCOME_TYPE"].value_counts().index, palette="Set3")
plt.title("Income Type Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "income_type_count.png"), dpi=110)
plt.close()

# 4. Annual income distribution
plt.figure(figsize=(6, 4))
sns.histplot(app_df["AMT_INCOME_TOTAL"], bins=40, kde=True, color="#1565C0")
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "income_distribution.png"), dpi=110)
plt.close()

# 5. Education level vs target
plt.figure(figsize=(8, 4))
edu_target = app_df.merge(proc_df[["TARGET"]], left_index=True, right_index=True) \
    if len(app_df) == len(proc_df) else None
sns.countplot(y="NAME_EDUCATION_TYPE", data=app_df,
              order=app_df["NAME_EDUCATION_TYPE"].value_counts().index, palette="coolwarm")
plt.title("Education Level Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "education_count.png"), dpi=110)
plt.close()

print("EDA plots saved to", OUT)
