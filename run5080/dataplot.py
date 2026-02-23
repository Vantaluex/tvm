import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# SCRIPT SETTINGS
CSV_FILE = "completed_datasets/gpu_dataset_r50x20000.csv"  
POWER_THRESHOLD = 40.0 
strPOWER_THRESHOLD = str(POWER_THRESHOLD)

# ==================================================
# GRAPH AXIS SETTINGS
X_MIN = -1.0      # Minimum Latency (ms)
X_MAX = None   # Maximum Latency (ms)
Y_MIN = 0.0      # Minimum Power (W)
Y_MAX = None   # Maximum Power (W)


print(f"[INFO] Loading dataset from: {CSV_FILE}")
df = pd.read_csv(CSV_FILE)

# Split the data into "normal" and "anomalies"
normal_df = df[df['avg_power_w'] >= POWER_THRESHOLD]
anomaly_df = df[df['avg_power_w'] < POWER_THRESHOLD]

print(f"[INFO] Total records loaded: {len(df)}")
print(f"[INFO] Normal records (>= {POWER_THRESHOLD}W): {len(normal_df)}")
print(f"[INFO] Anomalies (< {POWER_THRESHOLD}W): {len(anomaly_df)}")

# Print out the top 15 anomalies so you can inspect them in the terminal
if len(anomaly_df) > 0:
    print("\n[WARN] Sample Low-Power Anomalies:")
    print(anomaly_df[['i', 'workload_hash', 'lat_mean_ms', 'avg_power_w']].head(15).to_string(index=False))

# Create the Latency vs Power scatter plot
plt.figure(figsize=(10, 6))

# Plot normal data as blue dots
plt.scatter(normal_df['lat_mean_ms'], normal_df['avg_power_w'], 
            alpha=0.5, label='Normal (>= ' + strPOWER_THRESHOLD + 'W)', color='blue', s=10)

# Plot anomalies as red dots
plt.scatter(anomaly_df['lat_mean_ms'], anomaly_df['avg_power_w'], 
            alpha=0.7, label='Anomaly (< ' + strPOWER_THRESHOLD + 'W)', color='red', s=15)

# Add the horizontal red line indicating the anomaly threshold
plt.axhline(y=POWER_THRESHOLD, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Anomaly Threshold')

# Add the horizontal red line indicating the anomaly threshold
plt.axhline(y=18.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, label='idle power')

# Apply custom axis limits if they are defined
if X_MIN is not None or X_MAX is not None:
    plt.xlim(left=X_MIN, right=X_MAX)
if Y_MIN is not None or Y_MAX is not None:
    plt.ylim(bottom=Y_MIN, top=Y_MAX)

# Labeling the graph
plt.title('Latency vs Power for ResNet50 Schedules')
plt.xlabel('Latency (ms)')
plt.ylabel('Average Power (W)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the graph to an image file
output_graph = 'plotted_datas/lvp_full_r50.png'
plt.savefig(output_graph, dpi=300)
print(f"\n[INFO] Graph saved as '{output_graph}'")
