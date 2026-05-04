# Credit‑Risk‑Scorecard‑Engine
A machine‑learning pipeline that trains a logistic‑regression PD model, converts PD into a standardized credit score, extracts a human‑readable scorecard, visualizes performance, and validates that manual scoring matches model scoring.

ROC‑AUC: 0.866

**Overview**

This project builds a probability‑of‑default (PD) credit‑risk model and scorecard using:

– Logistic Regression with balanced class weights

– Scikit‑learn preprocessing (imputation, scaling, one‑hot encoding)

– Log‑odds PD -> Score transformation (300–900 range)

– Scorecard coefficient extraction

– Matplotlib visualizations

– Manual score reconstruction for verification

The engine automatically:

– Loads and preprocesses borrower and loan features

– Trains a PD model on a stratified train/test split

– Converts PD into a credit score using industry‑standard log‑odds mapping

– Segments borrowers into score bands and computes bad‑rate statistics

– Extracts numeric and categorical scorecard points

– Validates that manual scoring exactly matches model scoring

– Generates diagnostic plots (ROC, calibration, score distributions, score bands, PD→score curve, feature contributions)

**Potential Future Upgrades**

– Deploy as an API endpoint

– Add reject inference for more robust PD modeling

– Integrate with loan decisioning workflows

– Add dashboard (Streamlit / Dash)
