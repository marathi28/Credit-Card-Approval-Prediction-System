# Credit Card Approval Prediction System

A machine learning project that predicts whether a credit card application
will be **approved** or **rejected**, built with Python, Flask,
Scikit-learn, and XGBoost, with an optional IBM Watson Machine Learning
cloud deployment path.

This repository follows the standard AI-ML-and-GEN-AI-Track project
submission structure:

| Folder | Contents |
|---|---|
| [1. Brainstorming & Ideation](./1.%20Brainstorming%20%26%20Ideation) | Problem statement, approach, target scenarios |
| [2. Requirement Analysis](./2.%20Requirement%20Analysis) | Functional, non-functional & data requirements |
| [3. Project Design Phase](./3.%20Project%20Design%20Phase) | Architecture, data flow, module & UI design |
| [4. Project Planning Phase](./4.%20Project%20Planning%20Phase) | Timeline, task breakdown, milestones, risks |
| [5. Project Development Phase](./5.%20Project%20Development%20Phase) | Full working codebase (data, model, Flask app) |
| [6. Project Testing](./6.%20Project%20Testing) | Automated test suite + test report |
| [7. Project Documentation](./7.%20Project%20Documentation) | Full setup & usage README |
| [8. Project Demonstration](./8.%20Project%20Demonstration) | Walkthrough & demo script |

## Quick Start

```bash
cd "5. Project Development Phase"
pip install -r requirements.txt
python app.py
```
Open **http://127.0.0.1:5000**

For full details, see [`7. Project Documentation/README.md`](./7.%20Project%20Documentation/README.md).

## At a Glance

- **4 classification algorithms** trained and compared: Logistic Regression,
  Decision Tree, Random Forest, XGBoost
- **Best model selected automatically** by ROC-AUC and deployed to a live
  Flask web application
- **Real-time predictions** through a web form, with a risk-probability
  gauge and plain-language decision factors
- **Optional cloud deployment** to IBM Watson Machine Learning
