# 6. Project Testing

## Test Strategy

Testing covers three layers:
1. **Model evaluation** — accuracy, confusion matrix, classification report,
   ROC-AUC for each of the 4 trained classifiers (see `model/model_comparison.json`
   in the Development phase).
2. **Application/integration testing** — every Flask route, and the full
   request → prediction → response cycle (`test_app.py` in this folder).
3. **Behavioral sanity checks** — feeding intentionally strong and
   intentionally weak applicant profiles and confirming the prediction
   direction makes sense.

## Model Evaluation Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 76.08% | **0.832** ← selected |
| Random Forest | 82.92% | 0.810 |
| XGBoost | 78.25% | 0.791 |
| Decision Tree | 75.42% | 0.764 |

**Selection metric:** ROC-AUC, not raw accuracy — the approval/rejection
target is naturally imbalanced (~82% approved / ~18% rejected in the
training data), so a model can score high accuracy just by predicting the
majority class. ROC-AUC better reflects real discriminative ability, which
is why Logistic Regression (highest AUC) was promoted to production even
though Random Forest had marginally higher raw accuracy.

*(Exact figures may shift slightly on re-runs if the dataset is
regenerated with a different random seed or a real dataset is substituted —
re-run `model/train_models.py` to refresh `model_comparison.json`.)*

## Automated Test Suite (`test_app.py`)

Run with:
```bash
python "6. Project Testing/test_app.py"
```

| Test | Purpose | Result |
|---|---|---|
| `test_routes_return_200` | Home, prediction form, and model ledger pages all load | ✅ PASS |
| `test_prediction_returns_valid_result` | POST to `/predict` returns a valid Approved/Declined page | ✅ PASS |
| `test_strong_profile_trends_approved` | High income, owns assets, stable employment, no dependents → Approved | ✅ PASS |
| `test_weak_profile_trends_declined` | Low income, no assets, unemployed, several dependents → Declined | ✅ PASS |
| `test_model_artifacts_loaded` | Model, scaler, encoders, and feature list load without error | ✅ PASS |

**Latest run output:**
```
PASS: all GET routes return 200
PASS: prediction endpoint returns a valid Approved/Declined result
PASS: strong applicant profile -> Approved
PASS: weak applicant profile -> Declined
PASS: model artifacts loaded (best model = Logistic Regression)

All tests passed.
```

## Manual / Exploratory Testing

- Verified the prediction form's dropdown values always match what the
  label encoders were trained on (sourced directly from `label_encoders.pkl`),
  so no unseen-category errors are possible from the UI.
- Verified the risk-probability gauge visually reflects the returned
  probability (0–100%) and the decision "stamp" color/label matches the
  underlying binary prediction.
- Verified the Jupyter notebook (`5. Project Development Phase/notebook/`)
  executes top-to-bottom without errors and reproduces the same saved model
  artifacts used by the Flask app.

## Known Limitations

- The dataset used for training/testing is synthetic (schema-matched to the
  real-world Credit Card Approval dataset) since no production dataset was
  supplied — reported metrics reflect performance on synthetic data and
  should be re-validated once real applicant data is substituted.
- Explanatory "factors" shown on the result page are rule-based heuristics
  for readability, not full SHAP/LIME model-internal explanations.
