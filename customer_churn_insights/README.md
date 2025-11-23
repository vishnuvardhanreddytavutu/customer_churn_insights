# AI Customer Churn & Insights System

Predict customer churn from typical customer attributes and usage metrics. Includes feature engineering,
model training (XGBoost), explainability with SHAP, segmentation module, and Streamlit dashboard for insights.

## Structure
- data/telco_sample.csv      (synthetic sample dataset modeled after common telco churn schemas)
- train.py                   (trains XGBoost model, saves models/churn_model.joblib)
- explain.py                 (generate SHAP explainability artifacts)
- app.py                     (Streamlit dashboard for upload, predictions, and SHAP plots)
- requirements.txt
- Dockerfile
- LICENSE

## Quick start
pip install -r requirements.txt
python train.py
streamlit run app.py
