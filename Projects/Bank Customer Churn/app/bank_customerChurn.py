import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "../models/churn_model.pkl")
scaler_path = os.path.join(BASE_DIR, "../models/scaler.pkl")

model = joblib.load(model_path)

st.title("Bank Customer Churn Prediction")
st.write("Predict whether a customer will leave the bank or not based on their features.")

# Side bar for input features
"""'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'BalanceSalaryRatio', 'ProductDensity', 'EngagementScore',
    'AgeTenureRatio'
    
    These are my column names for the input data, and I will use these names to create a DataFrame for the input data.
"""
st.sidebar.header("Input Features")
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=600)
Geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.sidebar.number_input("Tenure", min_value=0, max_value=10, value=5)
balance = st.sidebar.number_input("Balance", min_value=0.0, value=10000.0)
num_of_products = st.sidebar.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.sidebar.selectbox("Has Credit Card", ["Yes", "No"])    
is_active_member = st.sidebar.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, value=50000.0)
BalanceSalaryRatio = balance / estimated_salary if estimated_salary > 0 else 0
ProductDensity = num_of_products / tenure if tenure > 0 else 0
EngagementScore = (BalanceSalaryRatio + ProductDensity) / 2
AgeTenureRatio = age / tenure if tenure > 0 else 0

# Convert categorical features to numerical
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0
Geography = 1 if Geography == "France" else (2 if Geography == "Spain" else 3)
gender = 1 if gender == "Male" else 0



# --- Create a DataFrame for the input data
data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [Geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'BalanceSalaryRatio': [BalanceSalaryRatio],
    'ProductDensity': [ProductDensity],
    'EngagementScore': [EngagementScore],
    'AgeTenureRatio': [AgeTenureRatio]
})

st.write("Input Data:")

# Display the input data
st.dataframe(data)


if st.button("Predict Churn"):
    prediction = model.predict(data)
    probability = model.predict_proba(data)
    st.subheader("Prediction Result")
    # give in green if the customer is likely to stay and red if the customer is likely to churn
    if prediction[0] == 1:
        st.markdown("<span style='color:red'>The customer is likely to churn.</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:green'>The customer is likely to stay.</span>", unsafe_allow_html=True)
    
    # Display the churn probability with two decimal places if the customer is likely to churn, otherwise display the probability of staying
    # if the customer is likely to churn, display the probability of churn, otherwise display the probability of staying
    # and give in red if the customer is likely to churn and green if the customer is likely to stay
    st.subheader("Churn Probability")
    st.write(f"Churn Probability: {probability[0][1]:.2f}")
