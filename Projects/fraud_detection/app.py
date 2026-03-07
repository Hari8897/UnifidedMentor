# ============================================================
# FRAUD TRANSACTION DETECTION STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Fraud Transaction Detection")
st.markdown("Upload multiple daily `.pkl` files to detect fraudulent transactions.")

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_final_model.pkl")

model = load_model()

# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Settings")

threshold = st.sidebar.slider(
    "Fraud Probability Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.05
)

uploaded_files = st.sidebar.file_uploader(
    "Upload .pkl files",
    type=["pkl"],
    accept_multiple_files=True
)

# ============================================================
# FEATURE ENGINEERING (FULL RULE COMPLIANT)
# ============================================================

def feature_engineering(df):

    if "TX_DATETIME" not in df.columns:
        df = df.reset_index()

    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    df = df.sort_values("TX_DATETIME")

    # --- Time Features
    df["tx_dayofweek"] = df["TX_DATETIME"].dt.dayofweek
    df["is_weekend"] = df["tx_dayofweek"].isin([5, 6]).astype(int)

    df["TX_TIME_DAYS"] = (
        df["TX_DATETIME"] - df["TX_DATETIME"].min()
    ).dt.days

    # --- Rule 1: High Amount
    df["high_amount_flag"] = (df["TX_AMOUNT"] > 220).astype(int)
    df["log_tx_amount"] = np.log1p(df["TX_AMOUNT"])

    # --- Rule 2: Terminal 28-Day Fraud
    df = df.sort_values(["TERMINAL_ID", "TX_DATETIME"])
    df = df.set_index("TX_DATETIME")

    df["term_fraud_count_28d"] = (
        df.groupby("TERMINAL_ID")["TX_FRAUD"]
          .rolling("28D")
          .sum()
          .reset_index(level=0, drop=True)
    )

    df["term_tx_count_28d"] = (
        df.groupby("TERMINAL_ID")["TX_AMOUNT"]
          .rolling("28D")
          .count()
          .reset_index(level=0, drop=True)
    )

    df = df.reset_index()

    df["term_fraud_rate_28d"] = (
        df["term_fraud_count_28d"] /
        (df["term_tx_count_28d"] + 1)
    )

    # --- Rule 3: Customer 14-Day Behavior
    df = df.sort_values(["CUSTOMER_ID", "TX_DATETIME"])
    df = df.set_index("TX_DATETIME")

    df["cust_avg_amt_14d"] = (
        df.groupby("CUSTOMER_ID")["TX_AMOUNT"]
          .rolling("14D")
          .mean()
          .reset_index(level=0, drop=True)
    )

    df["cust_tx_count_14d"] = (
        df.groupby("CUSTOMER_ID")["TX_AMOUNT"]
          .rolling("14D")
          .count()
          .reset_index(level=0, drop=True)
    )

    df = df.reset_index()

    df["cust_amt_ratio"] = (
        df["TX_AMOUNT"] /
        (df["cust_avg_amt_14d"] + 1)
    )

    df = df.fillna(0)

    FEATURES = [
        "TX_AMOUNT",
        "high_amount_flag",
        "log_tx_amount",
        "term_fraud_rate_28d",
        "term_fraud_count_28d",
        "cust_avg_amt_14d",
        "cust_tx_count_14d",
        "cust_amt_ratio",
        "TX_TIME_DAYS",
        "tx_dayofweek",
        "is_weekend"
    ]

    return df, df[FEATURES]

# ============================================================
# MAIN APP LOGIC
# ============================================================

if uploaded_files:

    df_list = []

    for file in uploaded_files:
        bytes_data = file.read()
        df_day = pd.read_pickle(io.BytesIO(bytes_data))
        df_list.append(df_day)

    df = pd.concat(df_list, ignore_index=True)
    df = df.reset_index(drop=True)

    st.success(f"Loaded {len(uploaded_files)} files")
    st.write("Total Transactions:", len(df))

    if st.button("🔍 Detect Fraud"):

        processed_df, X = feature_engineering(df)

        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        processed_df["Fraud_Probability"] = y_prob
        processed_df["Fraud_Prediction"] = y_pred

        total = len(processed_df)
        frauds = (processed_df["Fraud_Prediction"] == 1).sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", total)
        col2.metric("Frauds Detected", frauds)
        col3.metric("Legitimate", total - frauds)

        st.subheader("🚩 Fraudulent Transactions")

        fraud_df = processed_df[
            processed_df["Fraud_Prediction"] == 1
        ]

        if fraud_df.empty:
            st.warning("No fraud detected for selected threshold.")
        else:
            st.dataframe(
                fraud_df[
                    [
                        "TX_DATETIME",
                        "CUSTOMER_ID",
                        "TERMINAL_ID",
                        "TX_AMOUNT",
                        "Fraud_Probability"
                    ]
                ]
            )

else:
    st.info("Upload multiple `.pkl` files to begin detection.")
