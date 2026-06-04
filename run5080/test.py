from pathlib import Path
import re
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# USER SETTINGS
# ============================================================
#MODEL_NAME = "convnextv2-tiny"
#MODEL_NAME = "deberta-v3-base"
#MODEL_NAME = "deepseekr1-qwen-14b"
#MODEL_NAME = "densenet169"
#MODEL_NAME = "exaone-deep-7.8B"
#MODEL_NAME = "exaone3.5-7.8b"
#MODEL_NAME = "llama-3.1-8b"
#MODEL_NAME = "mask2former-swin-small"
#MODEL_NAME = "mobilenetv3large"
#MODEL_NAME = "modernbert-base"
#MODEL_NAME = "qwen2.5-3b"
#MODEL_NAME = "qwen2.5-9B"
#MODEL_NAME = "resnet50"
#MODEL_NAME = "rtdetr-r50" 
#MODEL_NAME = "segformer-b2" 
UNDERSCORE_NAME = MODEL_NAME.replace("-", "_")
DATASET_DIR = Path("/home/vantaluex/tvm19/run5080/completed_datasets")
MODEL_JSON = Path(f"/home/vantaluex/tvm19/run5080/midterm-report-output/version_B/model_split_suite/{UNDERSCORE_NAME}_model_split/B_{UNDERSCORE_NAME}_model_split_pooled.json")
OUTPUT_CSV = Path(f"/home/vantaluex/tvm19/run5080/plotted_datas/{MODEL_NAME}_all_freq_actual_vs_predicted.csv")
OUTPUT_PNG = Path(f"/home/vantaluex/tvm19/run5080/plotted_datas/{MODEL_NAME}_all_freq_prediction_accuracy.png")

# Set False if you want one unified cloud instead of per-frequency colors
COLOR_BY_FREQ = True

# ============================================================
# HELPERS
# ============================================================
def parse_freq_mhz(path: Path) -> float:
    m = re.search(r'@(\d+(?:\.\d+)?)x', path.name)
    if not m:
        raise ValueError(f"Could not parse frequency from filename: {path.name}")
    return float(m.group(1))

def pretty_model_name(name: str) -> str:
    return name.replace("_", "-")

# ============================================================
# FIND DATASETS
# ============================================================
pattern = f"dataset_{MODEL_NAME}@*.csv"
dataset_files = sorted(DATASET_DIR.glob(pattern))

if not dataset_files:
    raise FileNotFoundError(f"No files matched: {pattern}")

print("Matched dataset files:")
for p in dataset_files:
    print(" -", p.name)

# ============================================================
# LOAD MODEL
# ============================================================
booster = xgb.Booster()
booster.load_model(str(MODEL_JSON))

expected_features = booster.feature_names
if expected_features is None:
    raise ValueError("Booster has no feature_names metadata")

print("\nExpected feature count:", len(expected_features))
print("Last expected features:", expected_features[-5:])

# ============================================================
# PREDICT ALL FREQUENCIES
# ============================================================
parts = []

for csv_path in dataset_files:
    freq_mhz = parse_freq_mhz(csv_path)
    df = pd.read_csv(csv_path)

    if "avg_power_w" not in df.columns:
        raise ValueError(f"avg_power_w missing in {csv_path.name}")

    df["freq_mhz"] = freq_mhz

    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {missing}")

    X = df[expected_features].copy()
    dmatrix = xgb.DMatrix(X, feature_names=expected_features)
    pred = booster.predict(dmatrix)

    out = pd.DataFrame({
        "model": MODEL_NAME,
        "freq_mhz": freq_mhz,
        "actual_power": df["avg_power_w"],
        "predicted_power": pred,
    })

    for col in ["i", "workload_hash", "trace_hash", "lat_mean_ms", "n_stores"]:
        if col in df.columns:
            out[col] = df[col]

    out["source_csv"] = csv_path.name
    out["abs_error_w"] = (out["predicted_power"] - out["actual_power"]).abs()
    out["ape_pct"] = out["abs_error_w"] / out["actual_power"].clip(lower=1e-9) * 100.0
    parts.append(out)

combined = pd.concat(parts, ignore_index=True)
combined.to_csv(OUTPUT_CSV, index=False)

# ============================================================
# METRICS
# ============================================================
overall_mape = combined["ape_pct"].mean()
overall_mae = combined["abs_error_w"].mean()

per_freq = (
    combined.groupby("freq_mhz", as_index=False)
    .agg(
        mape_pct=("ape_pct", "mean"),
        mae_w=("abs_error_w", "mean"),
        count=("ape_pct", "size")
    )
    .sort_values("freq_mhz")
)

print("\nSaved CSV:", OUTPUT_CSV)
print("Rows:", len(combined))
print("Frequencies:", per_freq["freq_mhz"].tolist())
print("Overall MAPE (%):", overall_mape)
print("Overall MAE (W):", overall_mae)
print("\nPer-frequency metrics:")
print(per_freq.to_string(index=False))

# ============================================================
# PLOT
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

fig, ax = plt.subplots(figsize=(6.6, 6.0), dpi=300)

all_vals = pd.concat([combined["actual_power"], combined["predicted_power"]])
vmin = all_vals.min()
vmax = all_vals.max()
pad = max((vmax - vmin) * 0.04, 1.0)
lo = vmin - pad
hi = vmax + pad

freqs = per_freq["freq_mhz"].tolist()
palette = {
    2295.0: "#2b6cb0",
    2595.0: "#c53030",
    2617.0: "#c779d0",
}

if COLOR_BY_FREQ and len(freqs) > 1:
    for f in freqs:
        sub = combined[combined["freq_mhz"] == f]
        ax.scatter(
            sub["actual_power"],
            sub["predicted_power"],
            s=0.7,
            alpha=0.1,
            color=palette.get(f, None),
            edgecolors="none",
            rasterized=True,
            label=f"{int(f)} MHz",
        )
else:
    ax.scatter(
        combined["actual_power"],
        combined["predicted_power"],
        s=0.5,
        alpha=0.03,
        color="#c53030",
        edgecolors="none",
        rasterized=True,
    )

ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.9, color="gray", alpha=0.8)

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Actual Power (W)")
ax.set_ylabel("Predicted Power (W)")
ax.grid(True, alpha=0.15)

for spine in ax.spines.values():
    spine.set_alpha(0.3)

title = f"Prediction accuracy for {pretty_model_name(MODEL_NAME)}"
ax.set_title(title, pad=12)

summary = f"MAPE {overall_mape:.2f}% | MAE {overall_mae:.2f} W | N = {len(combined):,}"
fig.text(0.5, 0.94, summary, ha="center", va="center", fontsize=10)

if COLOR_BY_FREQ and len(freqs) > 1:
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor=palette.get(f, "C0"),
            markeredgecolor=palette.get(f, "C0"),
            label=f"{int(f)} MHz"
        )
        for f in freqs
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        edgecolor="0.85",
        fontsize=9,
    )

plt.tight_layout(rect=[0, 0, 1, 0.93])
OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PNG, bbox_inches="tight")
plt.show()

print("\nSaved PNG:", OUTPUT_PNG)