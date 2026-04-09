# Predictive Logistics Pipeline & MLOps Dashboard

[![Live Demo Hosted on Hugging Face](https://img.shields.io/badge/Live_Demo-Hugging_Face-blue?style=for-the-badge)](https://huggingface.co/spaces/danvin/logistics-prediction-engine)

## Project Overview
An end-to-end machine learning pipeline designed to predict supply chain delivery delays. The model moves beyond standard accuracy metrics by integrating a custom business cost-matrix, adjusting the prediction threshold to minimize financial loss from false negatives (missed delays). 

## Architecture & Tech Stack
* **Modeling:** XGBoost, `scikit-learn`.
* **Data Processing:** `pandas`, `numpy`, SMOTE (Isolated strictly to training data to prevent leakage).
* **Hyperparameter Tuning:** Optuna (Bayesian optimization).
* **Interpretability:** SHAP (TreeExplainer for local feature importance).
* **Deployment:** Gradio, Hugging Face Spaces.

## Business Logic Implementation
Standard classification models default to a 0.5 probability threshold. This project implements a financial penalty matrix:
* **False Positive Penalty:** $150 (Unnecessary expedited shipping)
* **False Negative Penalty:** $500 (Lost customer contract due to missed delay)
The algorithm calculates the exact probability threshold (typically ~0.30 to 0.40) that minimizes total corporate financial loss, demonstrating a focus on production-ready business metrics rather than isolated academic accuracy.

## Interpretability 
To ensure trust with warehouse operations teams, every inference generates a localized SHAP waterfall plot, explaining the exact factors (e.g., specific shipping lanes or supplier regions) driving the AI's decision.
