import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_disease_rf_model.pkl")

st.title("❤️ Heart Disease Prediction App")

with st.sidebar:    
    st.sidebar.header("Input Patient Data")
    
    # User inputs
    age = st.number_input("Age")
    sex = st.selectbox("Sex", [0,1])
    cp = st.selectbox("Chest Pain Type", [1,2,3,4])
    trestbps = st.number_input("Resting BP")
    chol = st.number_input("Cholesterol")
    fbs = st.selectbox("Fasting Blood Sugar", [0,1])
    restecg = st.selectbox("Rest ECG", [0,1,2])
    thalach = st.number_input("Max Heart Rate")
    exang = st.selectbox("Exercise Angina", [0,1])
    oldpeak = st.number_input("Oldpeak")
    slope = st.selectbox("ST Slope", [1,2,3])


# Show head of the input data for reference
st.subheader("Input Data Preview")
input_df = pd.DataFrame([[
    age, sex, cp, trestbps, chol, fbs, restecg, thalach,
    exang, oldpeak, slope
]], columns=[
    "age ","sex ","chest pain type ","resting bp s ","cholesterol ",
    "fasting blood sugar ","resting ecg ","max heart rate ",
    "exercise angina ","oldpeak ","ST slope "
])
st.write(input_df)    

if st.button("Predict"):
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol,
        fbs, restecg, thalach,
        exang, oldpeak, slope
    ]], columns=[
        "age","sex","chest pain type","resting bp s",
        "cholesterol","fasting blood sugar",
        "resting ecg","max heart rate",
        "exercise angina","oldpeak","ST slope"
    ])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"High Risk of Heart Disease ({prob:.2f})")
    else:
        st.success(f"Low Risk ({prob:.2f})")
