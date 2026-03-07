# ============================================================
# FRAUD DETECTION TRAINING (USING .PKL FILES)
# STRICTLY COMPLIANT WITH PROJECT RULES
# ============================================================

import pandas as pd
import numpy as np
import glob
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# ============================================================
# LOAD ALL .PKL FILES
# ============================================================

DATA_PATH = "data/"   # folder containing daily .pkl files

files = sorted(glob.glob(os.path.join(DATA_PATH, "*.pkl")))

if len(files) == 0:
    raise FileNotFoundError("No .pkl files found inside data/ folder")

df_list = []

for file in files:
    df_day = pd.read_pickle(file)
    df_list.append(df_day)

# Combine all days
df = pd.concat(df_list, ignore_index=False)

# Ensure datetime column
if "TX_DATETIME" not in df.columns:
    df = df.reset_index()

df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
df = df.sort_values("TX_DATETIME")

print("Total Transactions:", len(df))
print("Fraud Distribution:")
print(df["TX_FRAUD"].value_counts())

# ============================================================
# TIME FEATURES
# ============================================================

df["tx_hour"] = df["TX_DATETIME"].dt.hour
df["tx_day"] = df["TX_DATETIME"].dt.day
df["tx_dayofweek"] = df["TX_DATETIME"].dt.dayofweek
df["is_weekend"] = df["tx_dayofweek"].isin([5, 6]).astype(int)

df["TX_TIME_DAYS"] = (
    df["TX_DATETIME"] - df["TX_DATETIME"].min()
).dt.days

df["TX_TIME_SECONDS"] = (
    df["TX_DATETIME"].dt.hour * 3600 +
    df["TX_DATETIME"].dt.minute * 60 +
    df["TX_DATETIME"].dt.second
)

# ============================================================
# RULE 1: HIGH AMOUNT RULE
# ============================================================

df["high_amount_flag"] = (df["TX_AMOUNT"] > 220).astype(int)

# ============================================================
# RULE 2: TERMINAL 28-DAY FRAUD HISTORY (PAST ONLY)
# ============================================================

df["term_fraud_count_28d"] = (
    df.groupby("TERMINAL_ID")
      .rolling("28D", on="TX_DATETIME")["TX_FRAUD"]
      .sum()
      .reset_index(level=0, drop=True)
)

df["term_tx_count_28d"] = (
    df.groupby("TERMINAL_ID")
      .rolling("28D", on="TX_DATETIME")["TX_AMOUNT"]
      .count()
      .reset_index(level=0, drop=True)
)

df["term_fraud_rate_28d"] = (
    df["term_fraud_count_28d"] / (df["term_tx_count_28d"] + 1)
)

# ============================================================
# RULE 3: CUSTOMER 14-DAY BEHAVIOR (PAST ONLY)
# ============================================================

df["cust_avg_amt_14d"] = (
    df.groupby("CUSTOMER_ID")
      .rolling("14D", on="TX_DATETIME")["TX_AMOUNT"]
      .mean()
      .reset_index(level=0, drop=True)
)

df["cust_tx_count_14d"] = (
    df.groupby("CUSTOMER_ID")
      .rolling("14D", on="TX_DATETIME")["TX_AMOUNT"]
      .count()
      .reset_index(level=0, drop=True)
)

df["cust_amt_ratio"] = (
    df["TX_AMOUNT"] / (df["cust_avg_amt_14d"] + 1)
)

# ============================================================
# ADDITIONAL FEATURES
# ============================================================

df["log_tx_amount"] = np.log1p(df["TX_AMOUNT"])

df = df.fillna(0)

# ============================================================
# FINAL FEATURE LIST (NO LEAKAGE)
# ============================================================

FEATURES = [
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "tx_hour",
    "tx_day",
    "tx_dayofweek",
    "is_weekend",
    "high_amount_flag",
    "term_fraud_rate_28d",
    "cust_avg_amt_14d",
    "cust_tx_count_14d",
    "cust_amt_ratio",
    "log_tx_amount"
]

X = df[FEATURES]
y = df["TX_FRAUD"]

# ============================================================
# TIME-AWARE SPLIT (NO FUTURE LEAKAGE)
# ============================================================

split_date = df["TX_DATETIME"].quantile(0.8)

X_train = X[df["TX_DATETIME"] <= split_date]
y_train = y[df["TX_DATETIME"] <= split_date]

X_test  = X[df["TX_DATETIME"] > split_date]
y_test  = y[df["TX_DATETIME"] > split_date]

# ============================================================
# MODEL
# ============================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

pipeline.fit(X_train, y_train)

# ============================================================
# EVALUATION
# ============================================================

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(pipeline, "fraud_detection_pipeline_rule_compliant.pkl")

print("\nModel saved as fraud_detection_pipeline_rule_compliant.pkl")
