import pandas as pd
df = pd.read_csv("eyas_gpu_dataset_200.csv")

print("\n=== Cross-Task Power Spread ===")
task_means = df.groupby('workload_hash')['avg_power_w'].mean()
print(task_means)

print("\n=== Within-Task Variation (Eyas needs > 20% variance) ===")
for h, grp in df.groupby('workload_hash'):
    if len(grp) > 3:
        p_min = grp['avg_power_w'].min()
        p_max = grp['avg_power_w'].max()
        spread_pct = ((p_max - p_min) / p_max) * 100
        print(f"Hash: {h} | Count: {len(grp)} | Range: {p_min:.1f}W -> {p_max:.1f}W | Max variation: {spread_pct:.1f}%")
