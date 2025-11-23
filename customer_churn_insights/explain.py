import pandas as pd, joblib, shap
df = pd.read_csv('data/telco_sample.csv')
df['totalCharges'] = pd.to_numeric(df['totalCharges'], errors='coerce').fillna(0)
cat_cols = ['contract','onlineSecurity','techSupport','partner','dependents']
from sklearn.preprocessing import LabelEncoder
for c in cat_cols:
    df[c] = LabelEncoder().fit_transform(df[c])

X = df[['tenure','monthlyCharges','totalCharges'] + cat_cols + ['seniorCitizen']]
model = joblib.load('models/churn_model.joblib')

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
# save a small sample for app
import numpy as np
np.save('models/shap_sample.npy', shap_values[:50])
print('Saved shap_sample.npy')
