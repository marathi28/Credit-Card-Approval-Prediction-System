# 4. Project Planning Phase

## Project Timeline

| Phase | Task | Deliverable |
|---|---|---|
| 1. Brainstorming & Ideation | Define problem, approach, target users | Ideation document |
| 2. Requirement Analysis | Functional/non-functional/data requirements | Requirements document |
| 3. Project Design | Architecture, data flow, module & UI design | Design document + diagrams |
| 4. Project Planning | Task breakdown, milestones, risk plan | This document |
| 5. Project Development | Data pipeline, model training, Flask app | Working codebase |
| 6. Project Testing | Route tests, model evaluation, edge cases | Test report |
| 7. Project Documentation | README, setup guide, final report | Documentation set |
| 8. Project Demonstration | Walkthrough, screenshots, demo script | Demo materials |

## Task Breakdown (Development Phase)

1. Environment setup & package installation (NumPy, Pandas, Matplotlib,
   Seaborn, Scikit-learn, XGBoost, Flask, Joblib)
2. Dataset acquisition / generation & schema validation
3. Exploratory Data Analysis (count plots, distribution plots)
4. Data preprocessing & feature engineering
   - Handle missing values
   - Remove duplicates
   - Encode categorical variables
   - Convert payment status → binary target
5. Model training: Logistic Regression, Decision Tree, Random Forest, XGBoost
6. Model evaluation: accuracy, confusion matrix, classification report, ROC-AUC
7. Best model selection & artifact persistence (joblib)
8. Flask web application build: home, prediction form, model comparison
9. End-to-end testing of all routes and prediction flows
10. Optional: IBM Watson ML cloud deployment pipeline
11. Documentation & packaging for submission

## Milestones

- **M1 — Data ready:** cleaned, feature-engineered dataset available for training
- **M2 — Models trained:** all 4 classifiers trained and compared
- **M3 — App functional:** Flask app serves live predictions end-to-end
- **M4 — Documented & tested:** README, notebook, and test verification complete
- **M5 — Submission ready:** repository organized into the 8-phase template structure

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| No real dataset available at kickoff | Generate a synthetic, schema-accurate dataset so the pipeline is fully runnable; document how to swap in real data later |
| Class imbalance skews accuracy | Use `class_weight="balanced"` / `scale_pos_weight`, and prioritize ROC-AUC over accuracy when selecting the best model |
| Categorical values differ between training data and user input | Web form dropdowns are generated directly from the fitted `LabelEncoder` classes, so the UI can never submit an unseen category |
| Cloud deployment (Watson) requires paid IBM Cloud account | Made fully optional; local Flask app is a complete, standalone deliverable |

## Tools Used for Planning

Task breakdown and milestones tracked as this document; architecture diagrams
authored in Mermaid (Phase 3) for version-controlled, renderable planning
artifacts directly on GitHub.
