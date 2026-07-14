"""
deploy_to_watson.py
---------------------
Optional cloud deployment pipeline: publishes the trained best_model.pkl to
IBM Watson Machine Learning so predictions can be served from the cloud
instead of (or in addition to) the local Flask app.

SETUP
-----
1. pip install ibm-watson-machine-learning
2. Create a Watson Machine Learning service instance on IBM Cloud and an
   IBM Cloud API key: https://cloud.ibm.com/catalog/services/machine-learning
3. Fill in the credentials below (or better, set them as environment
   variables: WML_API_KEY, WML_URL, WML_SPACE_ID) rather than hardcoding.
4. Run: python deploy_to_watson.py

WHAT THIS SCRIPT DOES
----------------------
- Authenticates to Watson Machine Learning
- Packages the scikit-learn/XGBoost pipeline (scaler + model) so it can
  accept raw feature rows and return predictions
- Stores the model in your WML deployment space
- Creates an online deployment and prints the scoring endpoint URL

NOTE: This step requires an active IBM Cloud account and is optional —
the Flask app in this project works fully standalone without it.
"""

import os
import joblib
import numpy as np

try:
    from ibm_watson_machine_learning import APIClient
except ImportError:
    raise SystemExit(
        "ibm-watson-machine-learning is not installed.\n"
        "Run: pip install ibm-watson-machine-learning"
    )

# ---------------------------------------------------------------------
# 1. Credentials — prefer environment variables over hardcoding secrets
# ---------------------------------------------------------------------
API_KEY = os.environ.get("WML_API_KEY", "<YOUR_IBM_CLOUD_API_KEY>")
WML_URL = os.environ.get("WML_URL", "https://us-south.ml.cloud.ibm.com")
SPACE_ID = os.environ.get("WML_SPACE_ID", "<YOUR_DEPLOYMENT_SPACE_ID>")

wml_credentials = {"apikey": API_KEY, "url": WML_URL}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")


def main():
    if API_KEY.startswith("<") or SPACE_ID.startswith("<"):
        print("Set WML_API_KEY and WML_SPACE_ID (env vars) before running this script.")
        print("This is optional cloud-deployment tooling; the local Flask app does not need it.")
        return

    client = APIClient(wml_credentials)
    client.set.default_space(SPACE_ID)

    # ------------------------------------------------------------
    # 2. Load local artifacts produced by model/train_models.py
    # ------------------------------------------------------------
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

    with open(os.path.join(MODEL_DIR, "best_model_name.txt")) as f:
        model_name = f.read().strip()

    # ------------------------------------------------------------
    # 3. Wrap scaler + model into a single scikit-learn Pipeline so
    #    Watson ML can score raw (unscaled) feature rows directly.
    # ------------------------------------------------------------
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([("scaler", scaler), ("classifier", model)])

    # ------------------------------------------------------------
    # 4. Store the model in the deployment space
    # ------------------------------------------------------------
    sofware_spec_uid = client.software_specifications.get_id_by_name("runtime-23.1-py3.10")

    meta_props = {
        client.repository.ModelMetaNames.NAME: f"credit-card-approval-{model_name.lower().replace(' ', '-')}",
        client.repository.ModelMetaNames.TYPE: "scikit-learn_1.1",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sofware_spec_uid,
    }

    published_model = client.repository.store_model(
        model=pipeline, meta_props=meta_props,
        training_data=None, training_target=None,
    )
    model_uid = client.repository.get_model_id(published_model)
    print(f"Model stored in Watson ML. Model UID: {model_uid}")

    # ------------------------------------------------------------
    # 5. Create an online deployment
    # ------------------------------------------------------------
    deployment_meta = {
        client.deployments.ConfigurationMetaNames.NAME: "credit-card-approval-deployment",
        client.deployments.ConfigurationMetaNames.ONLINE: {},
    }
    deployment = client.deployments.create(model_uid, meta_props=deployment_meta)
    scoring_url = client.deployments.get_scoring_href(deployment)

    print("\nDeployment complete.")
    print("Scoring endpoint:", scoring_url)
    print("Feature order expected by the endpoint:", feature_columns)


if __name__ == "__main__":
    main()
