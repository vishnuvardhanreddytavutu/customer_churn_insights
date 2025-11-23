import pandas as pd, os, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

df = pd.read_csv('data/telco_sample.csv')
# basic preprocessing
df['totalCharges'] = pd.to_numeric(df['totalCharges'], errors='coerce').fillna(0)
cat_cols = ['contract','onlineSecurity','techSupport','partner','dependents']
for c in cat_cols:
    df[c] = LabelEncoder().fit_transform(df[c])

X = df[['tenure','monthlyCharges','totalCharges'] + cat_cols + ['seniorCitizen']]
y = (df['Churn']=='Yes').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
probs = model.predict_proba(X_test)[:,1]
print(classification_report(y_test, pred))
print('ROC AUC:', roc_auc_score(y_test, probs))

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/churn_model.joblib')
print('Saved models/churn_model.joblib')
