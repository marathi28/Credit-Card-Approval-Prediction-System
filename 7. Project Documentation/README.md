# Credit Card Approval Prediction System

A hands-on machine learning project that predicts whether a credit card
application will be **approved** or **rejected**, using Python, Flask,
scikit-learn/XGBoost, and an optional IBM Watson Machine Learning
deployment path.

> **Repository structure note:** this project is organized into the
> standard 8-phase submission template. The working codebase referenced
> below lives inside **`5. Project Development Phase/`**. Run all commands
> from inside that folder.

## What's included (inside `5. Project Development Phase/`)

```
5. Project Development Phase/
├── data/
│   ├── generate_data.py        # synthetic dataset generator (schema-matched)
│   ├── application_record.csv  # applicant demographic/financial data
│   ├── credit_record.csv       # monthly payment status history
│   ├── preprocess.py           # cleaning, feature engineering, encoding
│   ├── processed_data.csv      # model-ready dataset
│   └── eda.py                  # exploratory plots
├── notebook/
│   └── Credit_Card_Approval_EDA_and_Model_Training.ipynb
├── model/
│   ├── train_models.py         # trains & compares 4 classifiers
│   ├── best_model.pkl          # serialized winning model
│   ├── scaler.pkl              # fitted StandardScaler
│   ├── label_encoders.pkl      # fitted LabelEncoders per categorical field
│   ├── feature_columns.pkl     # exact feature order expected by the model
│   ├── best_model_name.txt
│   └── model_comparison.json   # accuracy / ROC-AUC for all 4 models
├── watson_deployment/
│   └── deploy_to_watson.py     # optional IBM Watson ML cloud deployment
├── templates/                  # Flask HTML templates
├── static/                     # CSS + generated EDA images
├── app.py                      # Flask web application
└── requirements.txt
```

Related phases elsewhere in this repository:
- **`6. Project Testing/`** — automated test suite (`test_app.py`) + test report
- **`8. Project Demonstration/`** — screenshots and demo walkthrough

## About the data

No dataset was attached to this brief, so `data/generate_data.py` produces a
**synthetic but schema-accurate** dataset that mirrors the well-known public
"Credit Card Approval Prediction" dataset structure (`application_record.csv`
+ `credit_record.csv`), including every feature named in the instructions:
Gender, Income Type, Annual Income, Employment Duration, Education Level,
Family Status, Housing Type, Age, Children, Family Members, and a multi-class
monthly payment `STATUS` history.

**To use your own real dataset instead:** drop `application_record.csv` and
`credit_record.csv` (same column names) into `data/`, then re-run:
```bash
python data/preprocess.py
python data/eda.py
python model/train_models.py
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate data (skip if you're supplying your own)
python data/generate_data.py

# 2. Preprocess & feature-engineer
python data/preprocess.py

# 3. Exploratory data analysis (saves plots to static/images/)
python data/eda.py

# 4. Train & compare all 4 models, save the best one
python model/train_models.py

# 5. Run the web app
python app.py
```
Then open **http://127.0.0.1:5000**.

Alternatively, open `notebook/Credit_Card_Approval_EDA_and_Model_Training.ipynb`
in Jupyter to walk through EDA, preprocessing, and model training interactively —
it saves the exact same artifacts consumed by `app.py`.

## Pipeline details

**1. Preprocessing & feature engineering** (`data/preprocess.py`)
- Merges applicant records with their payment history
- Converts the multi-class `STATUS` code into a binary target: an applicant
  is labeled high-risk (rejected) if they were ever 60+ days past due
- Handles missing values, removes duplicates
- Engineers `AGE_YEARS`, `EMPLOYMENT_YEARS`, `IS_CURRENTLY_EMPLOYED`
- Label-encodes all categorical fields

**2. Model training** (`model/train_models.py`)
Trains and compares:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

Each is evaluated with accuracy, confusion matrix, classification report, and
ROC-AUC. The model with the **highest ROC-AUC** (more meaningful than raw
accuracy on this imbalanced approve/reject task) is saved as the production
model.

**3. Flask web application** (`app.py`)
- `/` — project overview + EDA visuals
- `/predict` — application form → real-time approve/reject prediction with a
  risk-probability gauge and plain-language factors
- `/models` — side-by-side comparison of all 4 trained models

**4. Optional cloud deployment** (`watson_deployment/deploy_to_watson.py`)
Packages the scaler + model into a single scikit-learn `Pipeline` and
publishes it to an IBM Watson Machine Learning deployment space, printing a
live scoring endpoint. Requires an IBM Cloud account and API key — the local
Flask app works fully without this step.

## Retraining on new data

Because dropdown options in the web form and the label encoders are both
derived from the same `label_encoders.pkl`, retraining on a different
dataset (different income types, education levels, etc.) automatically
updates the UI — no template changes needed.
