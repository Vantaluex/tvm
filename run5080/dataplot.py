import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# SCRIPT SETTINGS
INPUT_DIR = "completed_datasets"
OUTPUT_DIR = "dataplots"
POWER_THRESHOLD = 20.0
IDLE_POWER = 16.0
FILE_PATTERN = "dataset_*.csv"
HIST_BINS = 60


# ==================================================
# SCATTER AXIS SETTINGS
X_MIN = -1.0
X_MAX = None
Y_MIN = 0.0
Y_MAX = None

# HISTOGRAM AXIS SETTINGS
POWER_X_MIN = 0.0
POWER_X_MAX = None
COUNT_Y_MIN = 0.0
COUNT_Y_MAX = None


# ==================================================
# HELPERS
def extract_frequency_group(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'@([^~]+)~', filename)
    if match:
        return match.group(1)
    return "unknown"


# ==================================================
# PREPARE OUTPUT DIRECTORY
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_paths = sorted(glob.glob(os.path.join(INPUT_DIR, FILE_PATTERN)))

print(f"[INFO] Input directory: {INPUT_DIR}")
print(f"[INFO] Output directory: {OUTPUT_DIR}")
print(f"[INFO] Files found: {len(csv_paths)}")

if not csv_paths:
    print("[WARN] No CSV files found. Check INPUT_DIR or FILE_PATTERN.")
    raise SystemExit(0)


# ==================================================
# STORE POWER VALUES PER FREQUENCY GROUP
freq_group_power = {}


# ==================================================
# PROCESS EACH DATASET FOR SCATTER PLOTS
for csv_file in csv_paths:
    print("\n" + "=" * 70)
    print(f"[INFO] Loading dataset from: {csv_file}")

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"[ERROR] Failed to read {csv_file}: {e}")
        continue

    required_cols = ['lat_mean_ms', 'avg_power_w']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"[ERROR] Skipping {csv_file} because required columns are missing: {missing_cols}")
        continue

    # Keep power samples for grouped histogram
    freq_group = extract_frequency_group(csv_file)
    if freq_group not in freq_group_power:
        freq_group_power[freq_group] = []

    freq_group_power[freq_group].extend(df['avg_power_w'].dropna().tolist())

    # Split for scatter plot
    normal_df = df[df['avg_power_w'] >= POWER_THRESHOLD]
    anomaly_df = df[df['avg_power_w'] < POWER_THRESHOLD]

    print(f"[INFO] Frequency group: {freq_group}")
    print(f"[INFO] Total records loaded: {len(df)}")
    print(f"[INFO] Normal records (>= {POWER_THRESHOLD}W): {len(normal_df)}")
    print(f"[INFO] Anomalies (< {POWER_THRESHOLD}W): {len(anomaly_df)}")

    if len(anomaly_df) > 0:
        print("\n[WARN] Sample Low-Power Anomalies:")
        sample_cols = [c for c in ['i', 'workload_hash', 'lat_mean_ms', 'avg_power_w'] if c in anomaly_df.columns]
        print(anomaly_df[sample_cols].head(15).to_string(index=False))

    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    scatter_output = os.path.join(OUTPUT_DIR, f"{base_name}_lvp.png")

    plt.figure(figsize=(10, 6))

    plt.scatter(
        normal_df['lat_mean_ms'],
        normal_df['avg_power_w'],
        alpha=0.5,
        label=f'Normal (>= {POWER_THRESHOLD}W)',
        color='blue',
        s=10
    )

    plt.scatter(
        anomaly_df['lat_mean_ms'],
        anomaly_df['avg_power_w'],
        alpha=0.7,
        label=f'Anomaly (< {POWER_THRESHOLD}W)',
        color='red',
        s=15
    )

    plt.axhline(
        y=POWER_THRESHOLD,
        color='red',
        linestyle='--',
        linewidth=1.5,
        alpha=0.8,
        label='Anomaly Threshold'
    )

    plt.axhline(
        y=IDLE_POWER,
        color='gray',
        linestyle='--',
        linewidth=1.5,
        alpha=0.8,
        label='Idle Power'
    )

    if X_MIN is not None or X_MAX is not None:
        plt.xlim(left=X_MIN, right=X_MAX)
    if Y_MIN is not None or Y_MAX is not None:
        plt.ylim(bottom=Y_MIN, top=Y_MAX)

    plt.title(f'Latency vs Power: {base_name}')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Average Power (W)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(scatter_output, dpi=300)
    plt.close()

    print(f"[INFO] Scatter plot saved as: {scatter_output}")


# ==================================================
# CREATE ONE HISTOGRAM PER FREQUENCY GROUP
print("\n" + "=" * 70)
print("[INFO] Creating combined histograms per frequency group...")

for freq_group, power_values in sorted(freq_group_power.items()):
    if not power_values:
        print(f"[WARN] No power values found for frequency group: {freq_group}")
        continue

    hist_output = os.path.join(OUTPUT_DIR, f"all_models_{freq_group}_power_hist.png")

    plt.figure(figsize=(10, 6))

    plt.hist(
        power_values,
        bins=HIST_BINS,
        color='steelblue',
        edgecolor='black',
        alpha=0.8
    )

    plt.axvline(
        x=POWER_THRESHOLD,
        color='red',
        linestyle='--',
        linewidth=1.5,
        alpha=0.8,
        label=f'Anomaly Threshold ({POWER_THRESHOLD}W)'
    )

    plt.axvline(
        x=IDLE_POWER,
        color='gray',
        linestyle='--',
        linewidth=1.5,
        alpha=0.8,
        label=f'Idle Power ({IDLE_POWER}W)'
    )

    if POWER_X_MIN is not None or POWER_X_MAX is not None:
        plt.xlim(left=POWER_X_MIN, right=POWER_X_MAX)
    if COUNT_Y_MIN is not None or COUNT_Y_MAX is not None:
        plt.ylim(bottom=COUNT_Y_MIN, top=COUNT_Y_MAX)

    plt.title(f'Power Distribution for All Models @ {freq_group}')
    plt.xlabel('Average Power (W)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(hist_output, dpi=300)
    plt.close()

    print(f"[INFO] Combined histogram saved as: {hist_output}")
    print(f"[INFO] Total samples in {freq_group}: {len(power_values)}")

print("\n[INFO] All datasets processed.")