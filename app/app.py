
# app.py -- Streamlit app to interact with the trained Loan Default model
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / 'models' / 'loan_model.joblib'

st.set_page_config(page_title='Loan Default Prediction', layout='centered')
st.title('Loan Default Prediction System')
st.write('Enter applicant details and get a probability of loan default.')

model = joblib.load(MODEL_PATH)

with st.form('input_form'):
    gender = st.selectbox('Gender', ['Male','Female','Prefer not to say'])
    married = st.selectbox('Married', ['Yes','No'])
    dependents = st.selectbox('Dependents', ['0','1','2','3+'])
    education = st.selectbox('Education', ['Graduate','Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes','No'])
    applicant_income = st.number_input('Applicant Income', min_value=0, value=25000, step=500)
    coapplicant_income = st.number_input('Coapplicant Income', min_value=0, value=0, step=500)
    loan_amount = st.number_input('Loan Amount', min_value=0, value=100000, step=1000)
    loan_amount_term = st.selectbox('Loan Term (months)', [360, 240, 180, 120])
    credit_history = st.selectbox('Credit History (1 = good)', [1.0, 0.0])
    property_area = st.selectbox('Property Area', ['Urban','Semiurban','Rural'])

    submitted = st.form_submit_button('Predict')

if submitted:
    input_df = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }])

    proba = model.predict_proba(input_df)[:,1][0]
    pred = int(proba >= 0.5)
    st.metric('Default probability', f'{proba:.2f}')
    st.write('Prediction:', 'Default' if pred==1 else 'No Default')
    st.write('Model suggests:', 'Reject loan' if pred==1 else 'Consider approval')
