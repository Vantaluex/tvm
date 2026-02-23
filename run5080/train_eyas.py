import pandas as pd
import numpy as np
import glob
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

print("[INFO] Initializing robust Eyas training pipeline...")

# 1. Robustness: Ensure output directory exists before doing any work
os.makedirs("trained_models", exist_ok=True)

CSV_FILE = "completed_datasets/gpu_dataset_r50x20000.csv"  
print(f"[INFO] Loading dataset from: {CSV_FILE}")
df = pd.read_csv(CSV_FILE)

# 2. Robustness: Check if all 656 feature columns actually exist
feature_cols = [f"f{k}" for k in range(656)]
missing = [c for c in feature_cols if c not in df.columns]
assert not missing, f"Missing feature columns in CSV: {missing[:5]}..."

X = df[feature_cols].values
y = df['avg_power_w'].values

# 3. Data Validity: Check for corrupted numbers (NaN or Infinity)
assert not np.any(np.isnan(y)) and not np.any(np.isinf(y)), "Fatal: Bad values (NaN/Inf) found in target power (y)."
assert not np.any(np.isinf(X)), "Fatal: Infinity (inf) found in feature columns (X)."
# XGBoost can handle NaNs in features, but we should log it
nan_count = np.isnan(X).sum()
if nan_count > 0:
    print(f"[WARN] Found {nan_count} NaNs in features. XGBoost sparsity-aware algorithm will handle them.")

print("[INFO] Computing stratification bins...")
# 4. Stratification Quality Check
bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
actual_bins = pd.Series(bins).nunique()
if actual_bins < 10:
    print(f"[WARN] Stratifying on {actual_bins}/10 bins. (Pandas dropped bins due to duplicate power values)")

# 5. The Three-Way Split: Train (80%), Val (10%), Test (10%)
# First, carve off 10% for the completely unseen Test set
X_temp, X_test, y_temp, y_test, bins_temp, _ = train_test_split(
    X, y, bins, test_size=0.10, stratify=bins, random_state=42
)

# Next, split the remaining 90% into Train and Validation
# We stratify again to ensure the Validation set has a good distribution
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1111, stratify=bins_temp, random_state=42 # 0.1111 of 90% is approx 10% of total
)

print(f"[INFO] Data Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

print("[INFO] Training XGBoost with Early Stopping...")
# 6. Model Definition with GPU Acceleration and Early Stopping capacity
model = xgb.XGBRegressor(
    n_estimators=2000,          
    max_depth=10,               
    learning_rate=0.05,
    subsample=0.8,              
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist',         
    device='cuda',
    early_stopping_rounds=50    # <-- MOVED HERE
)

# 7. Fit with Validation monitoring
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],  
    verbose=50                  # <-- REMOVED early_stopping_rounds from here
)

print(f"\n[INFO] Early Stopping finished training at Tree #{model.best_iteration}")

# 8. Overfitting Diagnostics: Train vs Test MAPE
print("[INFO] Evaluating model...")

# Move the model back to the CPU for inference to avoid the warning
# This exactly matches Eyas Section 6.3.3 (CPU inference)
model.set_params(device='cpu')

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# --- Sample predictions ---
results = pd.DataFrame({
    "Actual Power (W)": y_test[:10],
    "Predicted Power (W)": test_preds[:10],
})
results["Error (W)"] = (results["Actual Power (W)"] - results["Predicted Power (W)"]).abs()
print("\nSample Predictions:")
print(results.round(2).to_string(index=True))

# --- Optional: show the worst errors (useful for filtering/debugging) ---
abs_err = np.abs(y_test - test_preds)
worst_idx = np.argsort(abs_err)[-10:][::-1]  # top 10 worst
worst = pd.DataFrame({
    "Actual Power (W)": y_test[worst_idx],
    "Predicted Power (W)": test_preds[worst_idx],
    "Abs Error (W)": abs_err[worst_idx],
    "Pct Error (%)": abs_err[worst_idx] / y_test[worst_idx] * 100,
})
print("\nWorst 10 test-set errors:")
print(worst.round(2).to_string(index=False))


train_mape = mean_absolute_percentage_error(y_train, train_preds) * 100
test_mape = mean_absolute_percentage_error(y_test, test_preds) * 100

print("\n==================================")
print(f"Train MAPE: {train_mape:.2f}%")
print(f"Test MAPE:  {test_mape:.2f}%")
print("==================================\n")

print("\n[INFO] Saving model...")
model.save_model("trained_models/eyas_rtx5080_model_3.json")
print("[INFO] Done.")
