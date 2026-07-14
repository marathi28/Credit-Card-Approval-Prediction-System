# 1. Brainstorming & Ideation

## Problem Statement

Banks and financial institutions receive thousands of credit card applications
every day. A significant portion are rejected due to factors such as high
existing loan balances, insufficient income levels, or excessive credit
inquiries. Manually reviewing each application is time-consuming and prone to
human error, making it inefficient at scale.

## Core Idea

Automate the credit card approval decision using machine learning. Train a
predictive model on historical applicant data (financial + demographic
attributes) so the system can evaluate a new applicant and instantly predict
**Approved** or **Rejected** — the way a real bank underwriter would, but in
real time and at scale.

## Why Machine Learning?

| Manual Review | ML-Based System |
|---|---|
| Slow, one application at a time | Instant, real-time predictions |
| Inconsistent across reviewers | Consistent, repeatable decision logic |
| Hard to scale with volume | Scales to thousands of applications |
| Bias/fatigue risk | Decisions driven purely by data patterns |

## Brainstormed Approaches

1. **Rule-based system** (if income < X and loans > Y → reject) — rejected as
   an idea because real approval patterns are multi-dimensional and rules
   don't capture interactions between features.
2. **Single ML model** — simple but risks picking a model that isn't actually
   the best fit for this data.
3. **Compare multiple classification algorithms and pick the best one** —
   chosen approach. Train Logistic Regression, Decision Tree, Random Forest,
   and XGBoost side by side, evaluate objectively (accuracy + ROC-AUC), and
   deploy only the strongest performer.
4. **Deployment**: wrap the winning model in a Flask web app for a real,
   usable interface, with an optional IBM Watson ML cloud deployment path
   for scalable hosting.

## Target Users / Scenarios

- **Credit analysts** — screen new applications faster, prioritize manual
  review only where needed.
- **Compliance officers** — batch-screen applicants with past-due loan
  records and flag high-risk profiles.
- **Risk teams** — periodically re-score portfolio segments.
- **Prospective customers** — self-check eligibility before formally
  applying, reducing wasted paperwork and unnecessary rejections.

## Success Criteria

- Model achieves meaningfully-better-than-random discrimination between
  approved/rejected applicants (ROC-AUC target: > 0.75).
- Predictions are served in real time (< 1 second) through a web interface.
- The system is explainable enough that an analyst can see *why* a
  prediction leaned one way (key risk factors surfaced alongside the result).
