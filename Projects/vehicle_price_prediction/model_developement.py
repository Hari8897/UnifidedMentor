import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pipeline import build_pipeline   

# -----------------------------
# Load Dataset
# -----------------------------
script_dir = Path(__file__).resolve().parent
df = pd.read_csv(script_dir / "dataset.csv")

df = df.dropna(subset=["price"])

# Log-transform target
df["log_price"] = np.log1p(df["price"])

X = df.drop(columns=["price", "log_price", "name", "description"])
y = df["log_price"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# -----------------------------
# Build Pipeline
# -----------------------------
pipeline = build_pipeline(num_cols, cat_cols)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Hyperparameter Tuning
# -----------------------------
param_grid = {
    "model__n_estimators": [300, 500],
    "model__max_depth": [None, 20, 30],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt", "log2"]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=15,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

search.fit(X_train, y_train)
best_pipeline = search.best_estimator_

print("Best Params:", search.best_params_)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = best_pipeline.predict(X_test)

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE :", mean_absolute_error(y_test, y_pred))
print("R2  :", r2_score(y_test, y_pred))

# -----------------------------
# Save FINAL model (same folder)
# -----------------------------
model_path = script_dir / "vehicle_price_pipeline.pkl"
joblib.dump(best_pipeline, model_path)

print(f"✅ Model saved successfully at: {model_path}")

