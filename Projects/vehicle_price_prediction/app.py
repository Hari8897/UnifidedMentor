import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Vehicle Price Prediction",
    layout="wide"
)

st.title("🚗 Vehicle Price Prediction App")
st.write("Predict vehicle prices using a trained Machine Learning pipeline")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("vehicle_price_pipeline.pkl")

model = load_model()

# -----------------------------
# Load Dataset (for dropdowns)
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

df = load_data()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🔧 Vehicle Details")

year = st.sidebar.number_input(
    "Manufacturing Year",
    min_value=int(df["year"].min()),
    max_value=int(df["year"].max()),
    value=2023
)

mileage = st.sidebar.number_input(
    "Mileage",
    min_value=0,
    value=30000,
    step=1000
)


make = st.sidebar.selectbox("Make", sorted(df["make"].dropna().unique()))
model_name = st.sidebar.selectbox("Model", sorted(df["model"].dropna().unique()))
fuel = st.sidebar.selectbox("Fuel Type", sorted(df["fuel"].dropna().unique()))
transmission = st.sidebar.selectbox("Transmission", sorted(df["transmission"].dropna().unique()))
body = st.sidebar.selectbox("Body Type", sorted(df["body"].dropna().unique()))
drivetrain = st.sidebar.selectbox("Drivetrain", sorted(df["drivetrain"].dropna().unique()))

cylinders = st.sidebar.selectbox(
    "Cylinders",
    sorted(df["cylinders"].dropna().unique())
)

doors = st.sidebar.selectbox(
    "Doors",
    sorted(df["doors"].dropna().unique())
)

exterior_color = st.sidebar.selectbox(
    "Exterior Color",
    sorted(df["exterior_color"].dropna().unique())
)

interior_color = st.sidebar.selectbox(
    "Interior Color",
    sorted(df["interior_color"].dropna().unique())
)

engine = st.sidebar.selectbox(
    "Engine",
    sorted(df["engine"].dropna().unique())
)

trim = st.sidebar.selectbox(
    "Trim",
    sorted(df["trim"].dropna().unique())
)

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_df = pd.DataFrame({
    "year": [year],
    "mileage": [mileage],
    "make": [make],
    "model": [model_name],
    "fuel": [fuel],
    "transmission": [transmission],
    "body": [body],
    "drivetrain": [drivetrain],
    "cylinders": [cylinders],
    "doors": [doors],
    "exterior_color": [exterior_color],
    "interior_color": [interior_color],
    "engine": [engine],
    "trim": [trim]
})

st.subheader("📌 Input Summary")
st.dataframe(input_df)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Price"):
    log_price = model.predict(input_df)[0]
    predicted_price = np.expm1(log_price)

    st.success(f"💰 Estimated Vehicle Price: ${predicted_price:,.2f}")
