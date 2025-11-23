import streamlit as st
import pandas as pd, joblib, os
import plotly.express as px
st.title('AI Customer Churn & Insights System')

st.markdown('Upload a telco-like CSV or use the sample. Columns: tenure,monthlyCharges,totalCharges,contract,onlineSecurity,techSupport,seniorCitizen,partner,dependents')

uploaded = st.file_uploader('Upload CSV', type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info('Using bundled sample dataset.')
    df = pd.read_csv('data/telco_sample.csv')

st.dataframe(df.head())

if st.button('Train model (local)'):
    st.info('Training churn model... run python train.py in terminal for detailed output.')
    os.system('python train.py')

if st.button('Load model and predict churn'):
    try:
        model = joblib.load('models/churn_model.joblib')
    except Exception as e:
        st.error('Model not found. Run python train.py first.')
        st.stop()

    # simple preprocessing
    from sklearn.preprocessing import LabelEncoder
    df_proc = df.copy()
    df_proc['totalCharges'] = pd.to_numeric(df_proc['totalCharges'], errors='coerce').fillna(0)
    cat_cols = ['contract','onlineSecurity','techSupport','partner','dependents']
    for c in cat_cols:
        df_proc[c] = LabelEncoder().fit_transform(df_proc[c])
    X = df_proc[['tenure','monthlyCharges','totalCharges'] + cat_cols + ['seniorCitizen']]
    probs = model.predict_proba(X)[:,1]
    df['churn_prob'] = probs
    df['churn_pred'] = (df['churn_prob'] > 0.5).astype(int)
    st.dataframe(df[['customerID','churn_prob','churn_pred']].sort_values('churn_prob', ascending=False))

    fig = px.histogram(df, x='churn_prob', nbins=20, title='Churn probability distribution')
    st.plotly_chart(fig)

    st.markdown('Top risk segments (example):')
    seg = df[df['churn_prob']>0.6]
    if not seg.empty:
        st.table(seg[['customerID','contract','monthlyCharges','churn_prob']].head(10))
    else:
        st.markdown('No high-risk customers detected in sample.')

st.markdown('You can run explain.py after training to compute SHAP artifacts (optional).')
