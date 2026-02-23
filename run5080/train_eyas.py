import pandas as pd
import numpy as np
import os
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

os.makedirs("trained_models", exist_ok=True)

CSV_FILE = "completed_datasets/gpu_dataset_r50x20000.csv" # dataset path
MODEL_NAME = "resnet50_rtx5080_model_5" # output name
print(f"[INFO] Loading dataset from: {CSV_FILE}")
df = pd.read_csv(CSV_FILE)

feature_cols = [f"f{k}" for k in range(656)]
missing = [c for c in feature_cols if c not in df.columns]
assert not missing, f"Missing feature columns: {missing[:5]}..."

X = df[feature_cols].values
y = df['avg_power_w'].values

# Track the original row indices so we can perfectly map test data back to the DataFrame
indices = np.arange(len(df))

assert not np.any(np.isnan(y)) and not np.any(np.isinf(y)), "Bad values in y"
assert not np.any(np.isinf(X)), "Inf found in features"

print("[INFO] Computing stratification bins...")
bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
actual_bins = pd.Series(bins).nunique()

# --- Three-way split: 80% Train | 10% Val | 10% Test ---
# Pass `indices` through the split!
idx_temp, idx_test, X_temp, X_test, y_temp, y_test, bins_temp, _ = train_test_split(
    indices, X, y, bins, test_size=0.10, stratify=bins, random_state=42
)

# Re-quantize bins on the actual remaining y_temp distribution
bins_temp = pd.qcut(pd.Series(y_temp), q=actual_bins, labels=False, duplicates='drop')

# Pass `idx_temp` through the second split!
idx_train, idx_val, X_train, X_val, y_train, y_val = train_test_split(
    idx_temp, X_temp, y_temp, test_size=0.1111, stratify=bins_temp, random_state=42
)

print(f"[INFO] Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

print("[INFO] Training XGBoost on GPU with Early Stopping...")
model = xgb.XGBRegressor(
    n_estimators=2000,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist',
    device='cuda',
    early_stopping_rounds=50,
    eval_metric='mape'  # <-- FORCES early stopping to use MAPE instead of RMSE
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

print(f"\n[INFO] Early Stopping finished at Tree #{model.best_iteration}")

# Save then reload on CPU (Bypasses XGBoost set_params bug)
MODEL_PATH = "trained_models/"+ MODEL_NAME +".json"
model.save_model(MODEL_PATH)
cpu_model = xgb.XGBRegressor(device='cpu')
cpu_model.load_model(MODEL_PATH)

print("[INFO] Evaluating model on CPU...")
train_preds = cpu_model.predict(X_train)
val_preds   = cpu_model.predict(X_val)
test_preds  = cpu_model.predict(X_test)

train_mape = mean_absolute_percentage_error(y_train, train_preds) * 100
val_mape   = mean_absolute_percentage_error(y_val,   val_preds)   * 100
test_mape  = mean_absolute_percentage_error(y_test,  test_preds)  * 100

print("\n==================================")
print(f"Train MAPE: {train_mape:.2f}%")
print(f"Val MAPE:   {val_mape:.2f}%")
print(f"Test MAPE:  {test_mape:.2f}%")
print("==================================\n")

# --- Calculate Worst Indices ONCE (DRY principle) ---
abs_err = np.abs(y_test - test_preds)
worst_idx = np.argsort(abs_err)[-10:][::-1]

# 1. Sample Predictions Table
results = pd.DataFrame({
    "Actual Power (W)": y_test[:10],
    "Predicted Power (W)": test_preds[:10],
    "Error (W)": abs_err[:10]
})
print("Sample Predictions:")
print(results.round(2).to_string(index=True))

# 2. Worst 10 Table
worst = pd.DataFrame({
    "Actual Power (W)": y_test[worst_idx],
    "Predicted Power (W)": test_preds[worst_idx],
    "Abs Error (W)": abs_err[worst_idx],
    "Pct Error (%)": (abs_err[worst_idx] / y_test[worst_idx]) * 100,
})
print("\nWorst 10 test-set errors:")
print(worst.round(2).to_string(index=False))

# 3. Safely Map Back to DataFrame using Original Indices
print("\n[DIAGNOSTICS] True Workloads causing the worst errors:")
worst_original_idx = idx_test[worst_idx]

for i, (row_idx, pred, err) in enumerate(zip(worst_original_idx, test_preds[worst_idx], abs_err[worst_idx])):
    row = df.iloc[row_idx]
    print(f"Actual: {row['avg_power_w']:.1f}W | Pred: {pred:.1f}W | Err: {err:.1f}W | "
          f"Latency: {row['lat_mean_ms']:.2f}ms | Hash: {row['workload_hash']}")

print("\n[DIAGNOSTICS] Plotting Feature Importance...")
plt.figure(figsize=(10, 8))
xgb.plot_importance(cpu_model, max_num_features=20, importance_type='gain')
plt.title("Top 20 Features Predicting GPU Power (Gain)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
print("[INFO] Saved 'feature_importance.png'. Done.")



# code used to check suspicious schedules
# Inspect all rows with the repeat hash
suspect_full = df[df['workload_hash'] == 1479909360810603761].copy()
varying_feats = feat_std[feat_std > 1e-6].index.tolist()

# Pearson correlation of each varying feature with power
correlations = suspect_full[varying_feats].corrwith(suspect_full['avg_power_w'])
print("Top correlated features within this hash:")
print(correlations.abs().sort_values(ascending=False).head(20))

# Also check: is latency itself correlated with power here?
print(f"\nLatency-power correlation: {suspect_full['lat_mean_ms'].corr(suspect_full['avg_power_w']):.4f}")