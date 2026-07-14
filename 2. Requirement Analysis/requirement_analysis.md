# 2. Requirement Analysis

## Functional Requirements

| ID | Requirement |
|---|---|
| FR-1 | The system shall accept applicant financial & demographic data as input. |
| FR-2 | The system shall preprocess and clean raw applicant/credit history data. |
| FR-3 | The system shall train and compare at least 4 classification algorithms. |
| FR-4 | The system shall automatically select the best-performing model for deployment. |
| FR-5 | The system shall expose a web form for entering a new applicant's profile. |
| FR-6 | The system shall return an instant Approved/Rejected prediction with a risk probability. |
| FR-7 | The system shall display a model comparison view (accuracy, ROC-AUC per algorithm). |
| FR-8 | The system shall surface human-readable factors that influenced the decision. |
| FR-9 | The system shall optionally support cloud deployment via IBM Watson ML. |

## Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | Predictions must return in under 1 second for a single applicant. |
| NFR-2 | The web UI must be usable on desktop browsers (Chrome/Edge/Firefox). |
| NFR-3 | The trained model and preprocessing artifacts must be persisted (joblib) for reuse without retraining. |
| NFR-4 | The system must run on Windows/Linux/macOS with Python 3.8+. |
| NFR-5 | The codebase must be modular: data prep, training, and serving are separate, independently re-runnable scripts. |

## Data Requirements

Sourced from (or schema-matched to) the public **Credit Card Approval
Prediction** dataset structure, consisting of two linked files:

**`application_record.csv`** — one row per applicant:
- Gender, Owns Car, Owns Realty, Number of Children
- Annual Income, Income Type
- Education Level, Family Status, Housing Type
- Age (derived from DAYS_BIRTH), Employment Duration (derived from DAYS_EMPLOYED)
- Occupation Type, Family Member Count
- Contact flags: Work Phone, Phone, Email

**`credit_record.csv`** — one row per applicant per month:
- Monthly payment `STATUS` code (0–5 = days past due buckets, C = paid off, X = no loan)

These are merged and the multi-class `STATUS` history is converted into a
**binary target**: `0 = Approved` (good payment history) vs.
`1 = Rejected` (ever seriously delinquent, 60+ days past due).

## Constraints & Assumptions

- No production dataset was provided at project kickoff, so a synthetic,
  schema-accurate dataset was generated for development and demonstration
  (see `data/generate_data.py` in the Development phase). The pipeline is
  designed to be a drop-in replacement — swapping in a real dataset with the
  same column names requires no code changes downstream.
- Class imbalance is expected and typical of real approval data (most
  applicants are approved); models are evaluated with ROC-AUC in addition to
  accuracy to account for this.

## Skills / Technology Requirements

Python, Flask, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn,
Jupyter Notebook, IBM Watson Machine Learning (optional cloud deployment).

## System Requirements

**Hardware:** Intel i3 or above · 4 GB RAM minimum (8 GB recommended) ·
2 GB free disk space · internet connection for package installs and cloud
deployment.

**Software:** Windows / Linux / macOS · Python 3.8+ · Jupyter Notebook ·
VS Code or PyCharm · a modern web browser.
