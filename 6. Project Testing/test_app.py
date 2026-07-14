"""
test_app.py
------------
Basic automated tests for the Credit Card Approval Prediction System.
Run from the '5. Project Development Phase' folder (or adjust sys.path below):

    python "../6. Project Testing/test_app.py"

Covers:
- All Flask routes return HTTP 200
- The prediction endpoint returns a valid Approved/Declined result
- A clearly strong applicant profile is approved
- A clearly weak applicant profile is declined
- Model artifacts load without error and expose expected metadata
"""

import os
import sys

DEV_PHASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "5. Project Development Phase"
)
sys.path.insert(0, DEV_PHASE_DIR)
os.chdir(DEV_PHASE_DIR)

import app as flask_app_module  # noqa: E402


def test_routes_return_200():
    client = flask_app_module.app.test_client()
    for route in ["/", "/predict", "/models"]:
        resp = client.get(route)
        assert resp.status_code == 200, f"{route} returned {resp.status_code}"
    print("PASS: all GET routes return 200")


def _base_form():
    return {
        "gender": "M", "age": "35", "own_car": "Y", "own_realty": "Y",
        "children": "1", "fam_members": "3", "income": "250000",
        "income_type": flask_app_module.INCOME_TYPES[0],
        "education": flask_app_module.EDUCATION_TYPES[0],
        "family_status": flask_app_module.FAMILY_STATUSES[0],
        "housing_type": flask_app_module.HOUSING_TYPES[0],
        "occupation": flask_app_module.OCCUPATION_TYPES[0],
        "employment_years": "5", "is_employed": "1",
        "work_phone": "1", "phone": "1", "email": "1",
    }


def test_prediction_returns_valid_result():
    client = flask_app_module.app.test_client()
    resp = client.post("/predict", data=_base_form())
    assert resp.status_code == 200
    html = resp.data.decode()
    assert ("Approved" in html) or ("Declined" in html)
    print("PASS: prediction endpoint returns a valid Approved/Declined result")


def test_strong_profile_trends_approved():
    client = flask_app_module.app.test_client()
    form = _base_form()
    form.update({
        "income": "400000", "own_realty": "Y", "own_car": "Y",
        "children": "0", "is_employed": "1", "employment_years": "10",
        "income_type": "Commercial associate",
    })
    resp = client.post("/predict", data=form)
    html = resp.data.decode()
    assert "Approved" in html, "Expected a strong profile to be Approved"
    print("PASS: strong applicant profile -> Approved")


def test_weak_profile_trends_declined():
    client = flask_app_module.app.test_client()
    form = _base_form()
    form.update({
        "income": "60000", "own_realty": "N", "own_car": "N",
        "children": "4", "fam_members": "6", "is_employed": "0",
        "employment_years": "0", "income_type": "Pensioner", "age": "62",
    })
    resp = client.post("/predict", data=form)
    html = resp.data.decode()
    assert "Declined" in html, "Expected a weak profile to be Declined"
    print("PASS: weak applicant profile -> Declined")


def test_model_artifacts_loaded():
    assert flask_app_module.best_model is not None
    assert flask_app_module.scaler is not None
    assert len(flask_app_module.feature_columns) > 0
    assert flask_app_module.BEST_MODEL_NAME in flask_app_module.MODEL_COMPARISON
    print(f"PASS: model artifacts loaded (best model = {flask_app_module.BEST_MODEL_NAME})")


if __name__ == "__main__":
    test_routes_return_200()
    test_prediction_returns_valid_result()
    test_strong_profile_trends_approved()
    test_weak_profile_trends_declined()
    test_model_artifacts_loaded()
    print("\nAll tests passed.")
