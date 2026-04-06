import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ============================================================
# Version selection
# ============================================================

def parse_version():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python train_eyas.py A|B")

    version = sys.argv[1].strip().upper()
    if version not in {"A", "B"}:
        raise SystemExit("Usage: python train_eyas.py A|B")

    return version


VERSION = parse_version()


# ============================================================
# Global configuration
# ============================================================

DATASET_DIR = Path("completed_datasets")
EXCLUDE_PREFIX = "old_"

TARGET_COL = "avg_power_w"
LAT_COL = "lat_mean_ms"
WORKLOAD_HASH_COL = "workload_hash"
TRACE_HASH_COL = "trace_hash"
MODEL_COL = "model_name"
GROUP_ID_COL = "group_id"
FEATURE_COLS = [f"f{k}" for k in range(656)]

RANDOM_STATE = 42
TREE_METHOD = "hist"
DEPTH_GRID = [4, 6, 8, 9, 10, 11, 12]
MAX_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50
TEST_SIZE = 0.10
VAL_SIZE_FROM_TEMP = 1 / 9

BASE_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_lambda": 1.0,
    "tree_method": TREE_METHOD,
    "verbosity": 0,
    "seed": RANDOM_STATE,
    "disable_default_eval_metric": 1,
}


# ============================================================
# Dataset discovery
# ============================================================

def parse_model_name_from_path(path):
    stem = Path(path).stem
    if stem.startswith("dataset_"):
        stem = stem[len("dataset_"):]
    if "@" in stem:
        stem = stem.split("@", 1)[0]
    return stem


def discover_csv_files(dataset_dir=DATASET_DIR, exclude_prefix=EXCLUDE_PREFIX):
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    candidates = sorted(
        p for p in dataset_dir.glob("*.csv")
        if p.is_file() and not p.name.startswith(exclude_prefix)
    )

    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found in {dataset_dir} after excluding files starting with '{exclude_prefix}'"
        )

    csv_files = {}
    duplicates = {}

    for p in candidates:
        model_name = parse_model_name_from_path(p)
        if model_name in csv_files:
            duplicates.setdefault(model_name, [csv_files[model_name]])
            duplicates[model_name].append(str(p))
        else:
            csv_files[model_name] = str(p)

    if duplicates:
        lines = ["Duplicate model names discovered from filenames:"]
        for model_name, paths in sorted(duplicates.items()):
            lines.append(f"  - {model_name}")
            for path in paths:
                lines.append(f"      {path}")
        lines.append("Rename or remove duplicates so each parsed model name maps to exactly one CSV.")
        raise ValueError("\n".join(lines))

    return csv_files


def build_version_config(version):
    csv_files = discover_csv_files()

    if version == "A":
        return {
            "version": "A",
            "out_dir": Path("trained_models_A"),
            "csv_files": csv_files,
            "internal_split_mode": "row",
            "split_desc": "row-level stratified split (paper-style)",
            "full_desc": "This is in-distribution validation only, not unseen-model evaluation.",
        }

    return {
        "version": "B",
        "out_dir": Path("trained_models_B"),
        "csv_files": csv_files,
        "internal_split_mode": "grouped",
        "split_desc": "grouped split on model_name::workload_hash",
        "full_desc": "This uses grouped train/val splitting and is not unseen-model evaluation.",
    }


CFG = build_version_config(VERSION)

OUT_DIR = CFG["out_dir"]
LOG_DIR = OUT_DIR / "logs"
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


# ============================================================
# Logging
# ============================================================

class DualLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.fp = open(self.path, "w", encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def log(self, msg=""):
        print(msg)
        self.fp.write(str(msg) + "\n")
        self.fp.flush()

    def close(self):
        if not self.fp.closed:
            self.fp.close()


# ============================================================
# Helpers
# ============================================================

def sanitize_name(name):
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("@", "_at_")
        .replace("~", "_tilde_")
        .replace("-", "_")
    )


def detect_device():
    try:
        d = xgb.DMatrix(
            np.array([[0.0]], dtype=np.float32),
            label=np.array([0.0], dtype=np.float32),
        )
        xgb.train(
            {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "device": "cuda",
                "max_depth": 1,
                "verbosity": 0,
            },
            d,
            num_boost_round=1,
        )
        return "cuda"
    except Exception:
        return "cpu"


DEVICE = detect_device()


def safe_mape_array(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def safe_mape_xgb(preds, dmatrix):
    y_true = dmatrix.get_label()
    return "safe_mape", safe_mape_array(y_true, preds)


def make_power_bins(values, max_bins=10):
    s = pd.Series(values).astype(float)
    if s.nunique() < 2:
        return np.zeros(len(s), dtype=int)

    q = min(max_bins, int(s.nunique()))
    while q >= 2:
        try:
            bins = pd.qcut(s, q=q, labels=False, duplicates="drop")
            bins = np.asarray(bins, dtype=int)
            if len(np.unique(bins)) >= 2:
                return bins
        except Exception:
            pass
        q -= 1
    return np.zeros(len(s), dtype=int)


def validate_schema(df, model_name):
    required = ["i", WORKLOAD_HASH_COL, TRACE_HASH_COL, "n_stores", LAT_COL, TARGET_COL] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{model_name}] Missing columns: {missing[:10]}")


def load_one_dataset(model_name, csv_path, logger=None):
    if logger is not None:
        logger.log(f"[INFO] Loading {model_name} from: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_schema(df, model_name)

    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(
        subset=FEATURE_COLS + [TARGET_COL, LAT_COL, WORKLOAD_HASH_COL, TRACE_HASH_COL]
    ).copy()
    after = len(df)

    if after == 0:
        raise ValueError(f"[{model_name}] No valid rows remain after cleaning")

    df[MODEL_COL] = model_name
    df[GROUP_ID_COL] = df[MODEL_COL].astype(str) + "::" + df[WORKLOAD_HASH_COL].astype(str)

    if logger is not None:
        logger.log(f"[INFO] {model_name}: rows before cleaning = {before}")
        logger.log(f"[INFO] {model_name}: rows after  cleaning = {after}")
        logger.log(f"[INFO] {model_name}: dropped {before - after} rows with NaN/Inf issues")
        logger.log(f"[INFO] {model_name}: unique workloads = {df[WORKLOAD_HASH_COL].nunique()}")
        logger.log(f"[INFO] {model_name}: unique traces    = {df[TRACE_HASH_COL].nunique()}")
        logger.log(f"[INFO] {model_name}: unique groups    = {df[GROUP_ID_COL].nunique()}")

    return df


def load_all_datasets(csv_files, logger=None):
    frames = []
    for model_name, csv_path in csv_files.items():
        frames.append(load_one_dataset(model_name, csv_path, logger))

    full_df = pd.concat(frames, axis=0, ignore_index=True)

    if logger is not None:
        logger.log(f"[INFO] Total pooled rows: {len(full_df)}")
        logger.log(f"[INFO] Models: {sorted(full_df[MODEL_COL].unique().tolist())}")

    return full_df


def df_to_xy(df):
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)

    if np.any(np.isnan(X)):
        raise ValueError("NaN found in X")
    if np.any(np.isinf(X)):
        raise ValueError("Inf found in X")
    if np.any(np.isnan(y)):
        raise ValueError("NaN found in y")
    if np.any(np.isinf(y)):
        raise ValueError("Inf found in y")

    return X, y


# ============================================================
# Split logic
# ============================================================

def row_stratified_split(df, test_size, random_state, logger, split_name):
    bins = make_power_bins(df[TARGET_COL].values, max_bins=10)
    stratify_arg = bins if len(np.unique(bins)) >= 2 else None

    idx = np.arange(len(df))
    idx_a, idx_b = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    df_a = df.iloc[idx_a].copy().reset_index(drop=True)
    df_b = df.iloc[idx_b].copy().reset_index(drop=True)

    if logger is not None:
        logger.log(f"[INFO] {split_name}: row-level stratified split")
        logger.log(f"[INFO] {split_name}: rows A = {len(df_a)}")
        logger.log(f"[INFO] {split_name}: rows B = {len(df_b)}")
        logger.log(f"[INFO] {split_name}: strat bins = {len(np.unique(bins))}")

    return df_a, df_b


def grouped_stratified_split(df, test_size, random_state, logger, split_name):
    group_stats = (
        df.groupby(GROUP_ID_COL)
          .agg(
              group_power_mean=(TARGET_COL, "mean"),
              model_name=(MODEL_COL, "first"),
              workload_hash=(WORKLOAD_HASH_COL, "first"),
              n_rows=(TARGET_COL, "size"),
          )
          .reset_index()
    )

    n_groups = len(group_stats)
    if n_groups < 2:
        raise ValueError(f"{split_name}: need at least 2 groups, got {n_groups}")

    requested_test_groups = int(np.ceil(n_groups * test_size))
    max_bins = min(10, int(group_stats["group_power_mean"].nunique()))

    while max_bins >= 2:
        try:
            bins = pd.qcut(
                group_stats["group_power_mean"],
                q=max_bins,
                labels=False,
                duplicates="drop",
            )
            bins = np.asarray(bins, dtype=int)
            n_classes = len(np.unique(bins))

            if requested_test_groups >= n_classes:
                groups_a, groups_b = train_test_split(
                    group_stats[GROUP_ID_COL].values,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=bins,
                )

                df_a = df[df[GROUP_ID_COL].isin(groups_a)].copy().reset_index(drop=True)
                df_b = df[df[GROUP_ID_COL].isin(groups_b)].copy().reset_index(drop=True)

                overlap = set(df_a[GROUP_ID_COL].unique()).intersection(set(df_b[GROUP_ID_COL].unique()))
                if overlap:
                    raise ValueError(f"{split_name}: shared groups found")

                if logger is not None:
                    logger.log(f"[INFO] {split_name}: grouped split on {GROUP_ID_COL}")
                    logger.log(f"[INFO] {split_name}: total groups = {n_groups}")
                    logger.log(f"[INFO] {split_name}: requested test groups = {requested_test_groups}")
                    logger.log(f"[INFO] {split_name}: strat bins actually used = {n_classes}")
                    logger.log(f"[INFO] {split_name}: groups A = {df_a[GROUP_ID_COL].nunique()} | rows A = {len(df_a)}")
                    logger.log(f"[INFO] {split_name}: groups B = {df_b[GROUP_ID_COL].nunique()} | rows B = {len(df_b)}")
                    logger.log(f"[INFO] {split_name}: shared groups across split = {len(overlap)}")

                return df_a, df_b
        except Exception:
            pass

        max_bins -= 1

    groups_a, groups_b = train_test_split(
        group_stats[GROUP_ID_COL].values,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )

    df_a = df[df[GROUP_ID_COL].isin(groups_a)].copy().reset_index(drop=True)
    df_b = df[df[GROUP_ID_COL].isin(groups_b)].copy().reset_index(drop=True)

    overlap = set(df_a[GROUP_ID_COL].unique()).intersection(set(df_b[GROUP_ID_COL].unique()))
    if overlap:
        raise ValueError(f"{split_name}: shared groups found")

    if logger is not None:
        logger.log(f"[WARN] {split_name}: fell back to unstratified grouped split")
        logger.log(f"[INFO] {split_name}: total groups = {n_groups}")
        logger.log(f"[INFO] {split_name}: requested test groups = {requested_test_groups}")
        logger.log(f"[INFO] {split_name}: groups A = {df_a[GROUP_ID_COL].nunique()} | rows A = {len(df_a)}")
        logger.log(f"[INFO] {split_name}: groups B = {df_b[GROUP_ID_COL].nunique()} | rows B = {len(df_b)}")
        logger.log(f"[INFO] {split_name}: shared groups across split = {len(overlap)}")

    return df_a, df_b


def internal_split_fn(df, test_size, random_state, logger, split_name):
    if CFG["internal_split_mode"] == "row":
        return row_stratified_split(df, test_size, random_state, logger, split_name)
    return grouped_stratified_split(df, test_size, random_state, logger, split_name)


# ============================================================
# Training / evaluation
# ============================================================

def train_booster_with_safe_mape(dtrain, dval, params, num_boost_round, early_stopping_rounds, verbose_eval):
    evals = [(dtrain, "train"), (dval, "val")]
    try:
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            custom_metric=safe_mape_xgb,
            maximize=False,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )
    except TypeError:
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            feval=safe_mape_xgb,
            maximize=False,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )
    return booster


def predict_with_best_iteration(booster, X):
    d = xgb.DMatrix(X, feature_names=FEATURE_COLS)
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is not None:
        return booster.predict(d, iteration_range=(0, best_iteration + 1))
    return booster.predict(d)


def evaluate_split(booster, df, split_name, logger):
    X, y = df_to_xy(df)
    preds = predict_with_best_iteration(booster, X)

    mape = safe_mape_array(y, preds)
    mae = float(np.mean(np.abs(y - preds)))
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))

    logger.log(f"[INFO] {split_name} MAPE: {mape:.4f}%")
    logger.log(f"[INFO] {split_name} MAE : {mae:.4f} W")
    logger.log(f"[INFO] {split_name} RMSE: {rmse:.4f} W")

    return {"preds": preds, "mape": mape, "mae": mae, "rmse": rmse}


def compute_per_model_mape(df, preds):
    temp = df[[MODEL_COL, TARGET_COL]].copy().reset_index(drop=True)
    temp["pred_power_w"] = preds

    rows = []
    for model_name, part in temp.groupby(MODEL_COL):
        rows.append({
            "model_name": model_name,
            "rows": int(len(part)),
            "mape_pct": safe_mape_array(part[TARGET_COL].values, part["pred_power_w"].values),
        })

    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)


def log_test_diagnostics(df, preds, logger, title_prefix):
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    abs_err = np.abs(y - preds)

    logger.log("")
    logger.log(f"[DIAGNOSTICS] {title_prefix} sample predictions:")
    sample = pd.DataFrame({
        "model_name": df[MODEL_COL].values[:10],
        "actual_power_w": y[:10],
        "pred_power_w": preds[:10],
        "abs_error_w": abs_err[:10],
        "lat_mean_ms": df[LAT_COL].values[:10],
        "workload_hash": df[WORKLOAD_HASH_COL].values[:10],
        "trace_hash": df[TRACE_HASH_COL].values[:10],
    })
    logger.log(sample.round(4).to_string(index=False))

    worst_idx = np.argsort(abs_err)[-10:][::-1]
    worst = df.iloc[worst_idx].copy()
    worst["pred_power_w"] = preds[worst_idx]
    worst["abs_error_w"] = abs_err[worst_idx]
    worst["pct_error"] = (
        worst["abs_error_w"] /
        np.clip(np.abs(worst[TARGET_COL].to_numpy(dtype=np.float64)), 1e-8, None)
    ) * 100.0

    logger.log("")
    logger.log(f"[DIAGNOSTICS] {title_prefix} worst 10 rows:")
    logger.log(
        worst[
            [MODEL_COL, TARGET_COL, "pred_power_w", "abs_error_w", "pct_error", LAT_COL, WORKLOAD_HASH_COL, TRACE_HASH_COL]
        ].round(4).to_string(index=False)
    )


def save_feature_importance_plot(booster, out_png, title, logger):
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(booster, max_num_features=20, importance_type="gain")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    logger.log(f"[INFO] Saved feature-importance plot to: {out_png}")


def train_best_depth_model(train_df, val_df, model_tag, logger):
    X_train, y_train = df_to_xy(train_df)
    X_val, y_val = df_to_xy(val_df)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLS)

    logger.log(f"[INFO] Device selected: {DEVICE}")
    logger.log(f"[INFO] Depth grid: {DEPTH_GRID}")

    best_record = None
    best_booster = None

    for depth in DEPTH_GRID:
        logger.log("")
        logger.log("==========================================")
        logger.log(f"[INFO] Training candidate max_depth = {depth}")
        logger.log("==========================================")

        params = BASE_XGB_PARAMS.copy()
        params["max_depth"] = depth
        params["device"] = DEVICE

        booster = train_booster_with_safe_mape(
            dtrain=dtrain,
            dval=dval,
            params=params,
            num_boost_round=MAX_ESTIMATORS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=50,
        )

        val_preds = predict_with_best_iteration(booster, X_val)
        val_mape = safe_mape_array(y_val, val_preds)
        best_iteration = getattr(booster, "best_iteration", None)
        if best_iteration is None:
            best_iteration = MAX_ESTIMATORS - 1

        logger.log(f"[INFO] depth={depth} -> best_iteration={best_iteration}")
        logger.log(f"[INFO] depth={depth} -> validation safe_mape={val_mape:.4f}%")

        record = {
            "max_depth": depth,
            "best_iteration": int(best_iteration),
            "val_mape": float(val_mape),
        }

        if best_record is None or record["val_mape"] < best_record["val_mape"]:
            best_record = record
            best_booster = booster

    model_path = OUT_DIR / f"{model_tag}.json"
    best_booster.save_model(model_path)

    logger.log("")
    logger.log("##########################################")
    logger.log("[INFO] Best depth-search result")
    logger.log(f"[INFO] max_depth      = {best_record['max_depth']}")
    logger.log(f"[INFO] best_iteration = {best_record['best_iteration']}")
    logger.log(f"[INFO] best val MAPE  = {best_record['val_mape']:.4f}%")
    logger.log(f"[INFO] Saved model to  = {model_path}")
    logger.log("##########################################")

    return best_booster, best_record, model_path


# ============================================================
# Experiment 1: Leave-one-model-out
# ============================================================

def run_pooled_leave_one_model_out(full_df):
    held_out_models = sorted(full_df[MODEL_COL].unique().tolist())
    rows = []

    for fold_idx, held_out_model in enumerate(held_out_models, start=1):
        model_tag = f"{VERSION}_pooled_lomo_holdout_{sanitize_name(held_out_model)}"

        with DualLogger(LOG_DIR / f"{model_tag}.log") as logger:
            logger.log("============================================================")
            logger.log(f"[INFO] Version {VERSION} | Leave-One-Model-Out Fold {fold_idx}/{len(held_out_models)}")
            logger.log(f"[INFO] Held-out test model: {held_out_model}")
            logger.log(f"[INFO] Internal split style: {CFG['split_desc']}")
            if VERSION == "B":
                logger.log("[INFO] This is a conservative anti-leakage extension beyond Eyas.")
            logger.log("============================================================")

            test_df = full_df[full_df[MODEL_COL] == held_out_model].copy().reset_index(drop=True)
            train_pool_df = full_df[full_df[MODEL_COL] != held_out_model].copy().reset_index(drop=True)

            logger.log(f"[INFO] Training-pool models: {sorted(train_pool_df[MODEL_COL].unique().tolist())}")
            logger.log(f"[INFO] Test-set model only: {sorted(test_df[MODEL_COL].unique().tolist())}")
            logger.log(f"[INFO] Train-pool rows: {len(train_pool_df)}")
            logger.log(f"[INFO] Test rows      : {len(test_df)}")

            train_df, val_df = internal_split_fn(
                train_pool_df,
                test_size=0.10,
                random_state=RANDOM_STATE,
                logger=logger,
                split_name="LOMO train/val split",
            )

            booster, best_record, model_path = train_best_depth_model(train_df, val_df, model_tag, logger)

            logger.log("")
            logger.log("==================================")
            logger.log("[INFO] Fold evaluation")
            logger.log("==================================")

            train_metrics = evaluate_split(booster, train_df, "Train", logger)
            val_metrics = evaluate_split(booster, val_df, "Val", logger)
            test_metrics = evaluate_split(booster, test_df, "Test", logger)

            logger.log("")
            logger.log("[INFO] Held-out per-model MAPE table:")
            heldout_table = compute_per_model_mape(test_df, test_metrics["preds"])
            logger.log(heldout_table.round(4).to_string(index=False))

            log_test_diagnostics(test_df, test_metrics["preds"], logger, f"Version {VERSION} holdout={held_out_model}")

            plot_path = PLOT_DIR / f"{model_tag}_feature_importance.png"
            save_feature_importance_plot(
                booster,
                plot_path,
                f"Top 20 Features Predicting GPU Power ({VERSION} Holdout: {held_out_model})",
                logger,
            )

            rows.append({
                "fold": fold_idx,
                "held_out_model": held_out_model,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "best_depth": best_record["max_depth"],
                "best_iteration": best_record["best_iteration"],
                "train_mape_pct": train_metrics["mape"],
                "val_mape_pct": val_metrics["mape"],
                "test_mape_pct": test_metrics["mape"],
                "model_path": str(model_path),
                "plot_path": str(plot_path),
            })

    summary_df = pd.DataFrame(rows).sort_values("held_out_model").reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / "pooled_lomo_summary.csv", index=False)

    avg_row = {
        "fold": "AVG",
        "held_out_model": "ALL",
        "train_rows": summary_df["train_rows"].mean(),
        "val_rows": summary_df["val_rows"].mean(),
        "test_rows": summary_df["test_rows"].mean(),
        "best_depth": "",
        "best_iteration": summary_df["best_iteration"].mean(),
        "train_mape_pct": summary_df["train_mape_pct"].mean(),
        "val_mape_pct": summary_df["val_mape_pct"].mean(),
        "test_mape_pct": summary_df["test_mape_pct"].mean(),
        "model_path": "",
        "plot_path": "",
    }

    pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True).to_csv(
        OUT_DIR / "pooled_lomo_summary_with_avg.csv",
        index=False,
    )

    return summary_df


# ============================================================
# Experiment 2: Final full model
# ============================================================

def run_final_full_model(full_df):
    active_models = sorted(full_df[MODEL_COL].unique().tolist())
    num_models = len(active_models)

    model_tag = f"{VERSION}_full_all_models_{num_models}models"

    with DualLogger(LOG_DIR / f"{model_tag}.log") as logger:
        logger.log("============================================================")
        logger.log(f"[INFO] Version {VERSION} | Final full pooled model on all {num_models} discovered datasets")
        logger.log(f"[INFO] Active models: {active_models}")
        logger.log(f"[INFO] {CFG['full_desc']}")
        logger.log("============================================================")

        train_df, val_df = internal_split_fn(
            full_df,
            test_size=0.10,
            random_state=RANDOM_STATE,
            logger=logger,
            split_name="FULL pooled train/val split",
        )

        booster, best_record, model_path = train_best_depth_model(train_df, val_df, model_tag, logger)

        train_metrics = evaluate_split(booster, train_df, "Train", logger)
        val_metrics = evaluate_split(booster, val_df, "Val", logger)

        logger.log("")
        logger.log("[INFO] Validation per-model MAPE table:")
        logger.log(compute_per_model_mape(val_df, val_metrics["preds"]).round(4).to_string(index=False))

        plot_path = PLOT_DIR / f"{model_tag}_feature_importance.png"
        save_feature_importance_plot(
            booster,
            plot_path,
            f"Top 20 Features Predicting GPU Power ({VERSION} All {num_models} Models)",
            logger,
        )

    summary_df = pd.DataFrame([{
        "model_name": "ALL_MODELS_FULL",
        "num_models": num_models,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "best_depth": best_record["max_depth"],
        "best_iteration": best_record["best_iteration"],
        "train_mape_pct": train_metrics["mape"],
        "val_mape_pct": val_metrics["mape"],
        "model_path": str(model_path),
        "plot_path": str(plot_path),
    }])

    summary_df.to_csv(OUT_DIR / "full_model_summary.csv", index=False)
    return summary_df


# ============================================================
# Experiment 3: Per-model baselines
# ============================================================

def run_separate_per_model_baselines(full_df):
    rows = []

    for model_name in sorted(full_df[MODEL_COL].unique().tolist()):
        model_df = full_df[full_df[MODEL_COL] == model_name].copy().reset_index(drop=True)
        model_tag = f"{VERSION}_baseline_single_{sanitize_name(model_name)}"

        with DualLogger(LOG_DIR / f"{model_tag}.log") as logger:
            logger.log("============================================================")
            logger.log(f"[INFO] Version {VERSION} | Separate per-model baseline: {model_name}")
            logger.log(f"[INFO] Internal split style: {CFG['split_desc']}")
            logger.log("============================================================")

            temp_df, test_df = internal_split_fn(
                model_df,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                logger=logger,
                split_name=f"{model_name} temp/test split",
            )

            train_df, val_df = internal_split_fn(
                temp_df,
                test_size=VAL_SIZE_FROM_TEMP,
                random_state=RANDOM_STATE,
                logger=logger,
                split_name=f"{model_name} train/val split",
            )

            if VERSION == "B":
                train_groups = set(train_df[GROUP_ID_COL].unique())
                val_groups = set(val_df[GROUP_ID_COL].unique())
                test_groups = set(test_df[GROUP_ID_COL].unique())

                if not train_groups.isdisjoint(val_groups):
                    raise ValueError(f"{model_name}: train/val group overlap detected")
                if not train_groups.isdisjoint(test_groups):
                    raise ValueError(f"{model_name}: train/test group overlap detected")
                if not val_groups.isdisjoint(test_groups):
                    raise ValueError(f"{model_name}: val/test group overlap detected")

            booster, best_record, model_path = train_best_depth_model(train_df, val_df, model_tag, logger)

            train_metrics = evaluate_split(booster, train_df, "Train", logger)
            val_metrics = evaluate_split(booster, val_df, "Val", logger)
            test_metrics = evaluate_split(booster, test_df, "Test", logger)

            log_test_diagnostics(test_df, test_metrics["preds"], logger, f"Version {VERSION} per-model={model_name}")

            plot_path = PLOT_DIR / f"{model_tag}_feature_importance.png"
            save_feature_importance_plot(
                booster,
                plot_path,
                f"Top 20 Features Predicting GPU Power ({VERSION} {model_name})",
                logger,
            )

            rows.append({
                "model_name": model_name,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "best_depth": best_record["max_depth"],
                "best_iteration": best_record["best_iteration"],
                "train_mape_pct": train_metrics["mape"],
                "val_mape_pct": val_metrics["mape"],
                "test_mape_pct": test_metrics["mape"],
                "model_path": str(model_path),
                "plot_path": str(plot_path),
            })

    summary_df = pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / "per_model_baseline_summary.csv", index=False)
    return summary_df


# ============================================================
# Main
# ============================================================

def main():
    with DualLogger(LOG_DIR / "bootstrap.log") as boot_logger:
        boot_logger.log(f"[INFO] Bootstrapping Version {VERSION}...")
        boot_logger.log(f"[INFO] Device selected: {DEVICE}")
        boot_logger.log(f"[INFO] Dataset directory: {DATASET_DIR}")
        boot_logger.log(f"[INFO] Excluding files with prefix: {EXCLUDE_PREFIX}")
        boot_logger.log(f"[INFO] Internal split mode: {CFG['internal_split_mode']}")
        boot_logger.log(f"[INFO] Automatically discovered {len(CFG['csv_files'])} dataset files:")

        for model_name, csv_path in sorted(CFG["csv_files"].items()):
            boot_logger.log(f"    - {model_name}: {csv_path}")

        full_df = load_all_datasets(CFG["csv_files"], boot_logger)

    pooled_summary = run_pooled_leave_one_model_out(full_df)
    full_summary = run_final_full_model(full_df)
    baseline_summary = run_separate_per_model_baselines(full_df)

    print("\n============================================================")
    print(f"[INFO] VERSION {VERSION} COMPLETED")
    print(pooled_summary.round(4).to_string(index=False))
    print(full_summary.round(4).to_string(index=False))
    print(baseline_summary.round(4).to_string(index=False))
    print("============================================================\n")


if __name__ == "__main__":
    main()
