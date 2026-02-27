import pandas as pd
import numpy as np
import os
import xgboost as xgb
import matplotlib.pyplot as plt

# ==================================================
# SCRIPT SETTINGS
CSV_FILE = "completed_datasets/gpu_dataset_r50x20000.csv"  


print(f"[INFO] Loading dataset from: {CSV_FILE}")
df = pd.read_csv(CSV_FILE)

suspect_full = df[df['workload_hash'] == 1479909360810603761].copy()
suspect_full['energy_proxy'] = suspect_full['avg_power_w'] * suspect_full['lat_mean_ms']
# THEN slice
outliers = suspect_full[suspect_full['avg_power_w'] > 183.9]
normal   = suspect_full[suspect_full['avg_power_w'] <= 183.9]

print("Outlier schedules (high power):")
print(outliers[['avg_power_w', 'lat_mean_ms']].describe())

print("\nNormal schedules:")
print(normal[['avg_power_w', 'lat_mean_ms']].describe())

# Energy efficiency: lower is better (joules per operation)
suspect_full['energy_proxy'] = suspect_full['avg_power_w'] * suspect_full['lat_mean_ms']
print(f"\nEnergy proxy - outliers: {outliers['energy_proxy'].mean():.1f} | normal: {normal['energy_proxy'].mean():.1f}")
