import sys
import json
import random
import hashlib
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


# ============================================================
# CLI
# ============================================================

def parse_cli():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python train_eyas_rewrite_v3.py A|B [reuse|local_depth]\n"
            "  A/B           : split mode\n"
            "  reuse         : tune once on train-model partition, reuse downstream (default)\n"
            "  local_depth   : reuse train-model tuned params, optionally retune local depth neighborhood\n"
        )

    version = sys.argv[1].strip().upper()
    if version not in {"A", "B"}:
        raise SystemExit("First argument must be A or B")

    tuning_policy = "reuse"
    if len(sys.argv) >= 3:
        tuning_policy = sys.argv[2].strip().lower()
        if tuning_policy not in {"reuse", "local_depth"}:
            raise SystemExit("Second argument must be reuse or local_depth")

    return version, tuning_policy


VERSION, TUNING_POLICY = parse_cli()


# ============================================================
# Global config
# ============================================================

DATASET_DIR = Path("completed_datasets")
EXCLUDE_PREFIX = "old_"

TARGET_COL = "avg_power_w"
LAT_COL = "lat_mean_ms"
WORKLOAD_HASH_COL = "workload_hash"
TRACE_HASH_COL = "trace_hash"
MODEL_COL = "model_name"
GROUP_ID_COL = "group_id"
DOMAIN_COL = "training_domain"
TASK_FAMILY_COL = "task_family"

FEATURE_COLS = [f"f{k}" for k in range(656)]
EXPECTED_FEATURE_COUNT = 656

HARDWARE_CANDIDATE_COLS = [
    "hardware",
    "hardware_type",
    "device",
    "device_type",
    "platform",
    "target",
    "target_hw",
    "target_hardware",
]

RANDOM_STATE = 42
TREE_METHOD = "hist"

PAPER_TARGET_QUANTILE_BINS = 10
TEST_SIZE = 0.10
VAL_SIZE_FROM_TEMP = 1 / 9

MAX_ESTIMATORS = 3000
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 200

MAX_GLOBAL_SEARCH_TRIALS = 32
LOCAL_DEPTH_NEIGHBORHOOD = [-1, 0, 1]
MIN_ROWS_PER_MODEL_DOMAIN_BASELINE = 50

RUN_AUX_L0MO = False

BASE_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": TREE_METHOD,
    "verbosity": 0,
    "seed": RANDOM_STATE,
    "disable_default_eval_metric": 1,
}

SEARCH_SPACE = {
    "max_depth": [8, 9, 10, 11],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma": [0.0, 0.1],
    "reg_lambda": [1.0, 5.0],
    "reg_alpha": [0.0, 0.1],
    "min_child_weight": [1, 5],
}


# ============================================================
# Task-family mapping
# ============================================================

def normalize_model_key(name):
    key = str(name).strip().lower()
    for ch in ["/", "\\", " ", "-", ".", "@"]:
        key = key.replace(ch, "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_")


MODEL_TO_FAMILY = {
    # Vision classification
    "convnextv2_tiny": "vision_classification",
    "densenet169": "vision_classification",
    "mobilenetv3large": "vision_classification",
    "resnet50": "vision_classification",

    # Dense vision
    "rtdetr_r50": "vision_dense",
    "segformer_b2": "vision_dense",
    "mask2former_swin_small": "vision_dense",

    # Encoder-style text models
    "deberta_v3_base": "encoder_text",
    "modernbert_base": "encoder_text",

    # Decoder / generative LLMs
    "deepseekr1_qwen_14b": "decoder_llm",
    "exaone_deep_7_8b": "decoder_llm",
    "exaone3_5_7_8b": "decoder_llm",
    "llama_3_1_8b": "decoder_llm",
    "qwen2_5_3b": "decoder_llm",
    "qwen2_5_9b": "decoder_llm",
}


# ============================================================
# Version config
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
            f"No CSV files found in {dataset_dir} after excluding '{exclude_prefix}*'"
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
            "out_dir": Path("trained_models_A_rewrite_v3"),
            "csv_files": csv_files,
            "internal_split_mode": "row",
            "split_desc": "row-level stratified split",
        }

    return {
        "version": "B",
        "out_dir": Path("trained_models_B_rewrite_v3"),
        "csv_files": csv_files,
        "internal_split_mode": "grouped",
        "split_desc": "grouped split on model_name::workload_hash",
    }


CFG = build_version_config(VERSION)

OUT_DIR = CFG["out_dir"]
LOG_DIR = OUT_DIR / "logs"
META_DIR = OUT_DIR / "meta"

OUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
META_DIR.mkdir(exist_ok=True)


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
        line = str(msg) + "\n"
        print(msg)
        try:
            self.fp.write(line)
            self.fp.flush()
        except OSError as e:
            print(f"[LOGGER ERROR] Failed to write to log file: {e}")

    def close(self):
        if not self.fp.closed:
            self.fp.close()


# ============================================================
# Helpers
# ============================================================

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2)


def sanitize_name(name):
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("@", "_at_")
        .replace("-", "_")
        .replace(":", "_")
    )


def normalize_model_key(name):
    key = str(name).strip().lower()
    for ch in ["/", "\\", " ", "-", ".", "@"]:
        key = key.replace(ch, "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_")


def assign_task_family(model_name):
    key = normalize_model_key(model_name)
    if key not in MODEL_TO_FAMILY:
        raise KeyError(
            f"Model '{model_name}' missing from MODEL_TO_FAMILY. "
            "Add it explicitly for reproducible family-aware splitting."
        )
    return MODEL_TO_FAMILY[key]


def deterministic_seed_from_text(text, base_seed=RANDOM_STATE):
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
    return base_seed + (h % 100000)


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


def mae_array(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_array(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def detect_hardware_col(df):
    for col in HARDWARE_CANDIDATE_COLS:
        if col in df.columns:
            return col
    return None


def is_valid_stratify_labels(labels):
    if labels is None:
        return False
    s = pd.Series(labels)
    counts = s.value_counts(dropna=False)
    return len(counts) >= 2 and int(counts.min()) >= 2


def make_power_bins(values, max_bins=10, logger=None, context="split"):
    s = pd.Series(values).astype(float)

    if s.nunique() < 2:
        if logger is not None:
            logger.log(f"[WARN] {context}: label has <2 unique values; stratification collapsed to 1 bin")
        return np.zeros(len(s), dtype=int), 1

    q = min(max_bins, int(s.nunique()))

    while q >= 2:
        try:
            bins = pd.qcut(s, q=q, labels=False, duplicates="drop")
            bins = np.asarray(bins, dtype=int)
            if is_valid_stratify_labels(bins):
                if logger is not None and q < max_bins:
                    logger.log(f"[WARN] {context}: could only create {q} quantile bins instead of requested {max_bins}")
                return bins, len(np.unique(bins))
        except Exception:
            pass
        q -= 1

    if logger is not None:
        logger.log(f"[WARN] {context}: failed to create >=2 usable quantile bins; stratification collapsed to 1 bin")
    return np.zeros(len(s), dtype=int), 1


def validate_schema(df, model_name):
    required = ["i", WORKLOAD_HASH_COL, TRACE_HASH_COL, "n_stores", LAT_COL, TARGET_COL] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{model_name}] Missing columns: {missing[:20]}")
    if len(FEATURE_COLS) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Expected {EXPECTED_FEATURE_COUNT} feature columns, got {len(FEATURE_COLS)}")


def check_feature_aggregation_signature(df, logger, context):
    feat_df = df[FEATURE_COLS]

    n_zero_cols = int((feat_df == 0).all(axis=0).sum())
    n_negative_cols = int((feat_df < 0).any(axis=0).sum())
    n_constant_cols = int((feat_df.nunique(dropna=False) <= 1).sum())
    n_nan_cols = int(feat_df.isna().any(axis=0).sum())

    logger.log(f"[FEATURE CHECK] {context}: total feature columns = {len(FEATURE_COLS)}")
    logger.log(f"[FEATURE CHECK] {context}: all-zero columns      = {n_zero_cols}")
    logger.log(f"[FEATURE CHECK] {context}: columns w/ negatives  = {n_negative_cols}")
    logger.log(f"[FEATURE CHECK] {context}: constant columns      = {n_constant_cols}")
    logger.log(f"[FEATURE CHECK] {context}: columns w/ NaNs       = {n_nan_cols}")

    if len(FEATURE_COLS) % 4 == 0:
        logger.log(f"[FEATURE CHECK] {context}: feature count divisible by 4 = yes ({len(FEATURE_COLS) // 4} quartets)")
    else:
        logger.log(f"[FEATURE CHECK] {context}: feature count divisible by 4 = no")

    logger.log(
        f"[FEATURE CHECK] {context}: NOTE: this is a diagnostic signature only; "
        f"it cannot prove the upstream mean/std/min/max aggregation pipeline."
    )

    if n_zero_cols > 100:
        logger.log(f"[WARN] {context}: many all-zero feature columns; verify aggregation pipeline")
    if n_constant_cols > 100:
        logger.log(f"[WARN] {context}: many constant feature columns; discriminative signal may be weak")


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

    hw_col = detect_hardware_col(df)
    if hw_col is not None:
        df[DOMAIN_COL] = df[hw_col].astype(str)
    else:
        df[DOMAIN_COL] = "pooled_no_hardware_col"

    df[TASK_FAMILY_COL] = assign_task_family(model_name)

    if logger is not None:
        logger.log(f"[INFO] {model_name}: rows before cleaning = {before}")
        logger.log(f"[INFO] {model_name}: rows after cleaning  = {after}")
        logger.log(f"[INFO] {model_name}: task family          = {df[TASK_FAMILY_COL].iloc[0]}")
        logger.log(f"[INFO] {model_name}: domain source        = {hw_col if hw_col else 'MISSING'}")
        logger.log(f"[INFO] {model_name}: distinct domains     = {sorted(df[DOMAIN_COL].unique().tolist())}")
        check_feature_aggregation_signature(df, logger, context=model_name)

    return df


def load_all_datasets(csv_files, logger=None):
    frames = []
    for model_name, csv_path in csv_files.items():
        frames.append(load_one_dataset(model_name, csv_path, logger))

    full_df = pd.concat(frames, axis=0, ignore_index=True)

    if logger is not None:
        logger.log(f"[INFO] Total pooled rows: {len(full_df)}")
        logger.log(f"[INFO] Models: {sorted(full_df[MODEL_COL].unique().tolist())}")
        logger.log(f"[INFO] Families: {sorted(full_df[TASK_FAMILY_COL].unique().tolist())}")
        logger.log(f"[INFO] Domains: {sorted(full_df[DOMAIN_COL].unique().tolist())}")
        if full_df[DOMAIN_COL].nunique() == 1 and full_df[DOMAIN_COL].iloc[0] == "pooled_no_hardware_col":
            logger.log("[WARN] No hardware column detected. This prevents Eyas-style per-hardware training.")

    return full_df

def validate_family_inventory(df, logger):
    counts = (
        df.groupby(TASK_FAMILY_COL)[MODEL_COL]
        .nunique()
        .sort_values()
    )

    logger.log("[INFO] Model counts per family:")
    for family, n in counts.items():
        logger.log(f"  - {family}: {n}")

    too_small = counts[counts < 2]
    if len(too_small):
        raise ValueError(
            "Each family needs at least 2 models for unseen-model train/test splitting. "
            f"Too small: {too_small.to_dict()}"
        )

def df_to_xy(df):
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)

    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Invalid values found in X")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Invalid values found in y")
    return X, y


# ============================================================
# Split logic
# ============================================================

import math


def max_strat_bins_allowed(n_items, test_size):
    if n_items < 2:
        return 1

    if isinstance(test_size, float):
        n_test = math.ceil(n_items * test_size)
    else:
        n_test = int(test_size)

    n_train = n_items - n_test

    # For stratified splitting, both train and test need room for every class/bin.
    return max(1, min(PAPER_TARGET_QUANTILE_BINS, n_test, n_train))

def row_stratified_split(df, test_size, random_state, logger, split_name):
    allowed_bins = max_strat_bins_allowed(len(df), test_size)

    bins, n_bins = make_power_bins(
        df[TARGET_COL].values,
        max_bins=allowed_bins,
        logger=logger,
        context=split_name,
    )

    if logger is not None:
        logger.log(f"[INFO] {split_name}: allowed strat bins by split capacity = {allowed_bins}")
    stratify_arg = bins if is_valid_stratify_labels(bins) else None

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
        logger.log(f"[INFO] {split_name}: row-level split")
        logger.log(f"[INFO] {split_name}: rows A = {len(df_a)}")
        logger.log(f"[INFO] {split_name}: rows B = {len(df_b)}")
        logger.log(f"[INFO] {split_name}: strat bins used = {n_bins}")

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

    if len(group_stats) < 2:
        raise ValueError(f"{split_name}: need at least 2 groups")

    allowed_bins = max_strat_bins_allowed(len(group_stats), test_size)

    bins, n_bins = make_power_bins(
        group_stats["group_power_mean"].values,
        max_bins=allowed_bins,
        logger=logger,
        context=f"{split_name} (group means)",
    )

    if logger is not None:
        logger.log(
            f"[INFO] {split_name}: allowed strat bins by split capacity = {allowed_bins}"
        )

    stratify_arg = bins if is_valid_stratify_labels(bins) else None

    groups_a, groups_b = train_test_split(
        group_stats[GROUP_ID_COL].values,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    df_a = df[df[GROUP_ID_COL].isin(groups_a)].copy().reset_index(drop=True)
    df_b = df[df[GROUP_ID_COL].isin(groups_b)].copy().reset_index(drop=True)

    overlap = set(df_a[GROUP_ID_COL].unique()).intersection(set(df_b[GROUP_ID_COL].unique()))
    if overlap:
        raise ValueError(f"{split_name}: shared groups found across split")

    if logger is not None:
        logger.log(f"[INFO] {split_name}: grouped split on {GROUP_ID_COL}")
        logger.log(f"[INFO] {split_name}: groups A = {df_a[GROUP_ID_COL].nunique()} | rows A = {len(df_a)}")
        logger.log(f"[INFO] {split_name}: groups B = {df_b[GROUP_ID_COL].nunique()} | rows B = {len(df_b)}")
        logger.log(f"[INFO] {split_name}: strat bins used = {n_bins}")

    return df_a, df_b


def internal_split_fn(df, test_size, random_state, logger, split_name):
    if CFG["internal_split_mode"] == "row":
        return row_stratified_split(df, test_size, random_state, logger, split_name)
    return grouped_stratified_split(df, test_size, random_state, logger, split_name)


# ============================================================
# XGBoost training
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
    mae = mae_array(y, preds)
    rmse = rmse_array(y, preds)

    logger.log(f"[INFO] {split_name} MAPE: {mape:.4f}%")
    logger.log(f"[INFO] {split_name} MAE : {mae:.4f} W")
    logger.log(f"[INFO] {split_name} RMSE: {rmse:.4f} W")

    return {"preds": preds, "mape": mape, "mae": mae, "rmse": rmse}


def compute_per_model_mape(df, preds):
    temp = df[[MODEL_COL, TASK_FAMILY_COL, TARGET_COL]].copy().reset_index(drop=True)
    temp["pred_power_w"] = preds

    rows = []
    for model_name, part in temp.groupby(MODEL_COL):
        rows.append({
            "model_name": model_name,
            "task_family": part[TASK_FAMILY_COL].iloc[0],
            "rows": int(len(part)),
            "mape_pct": safe_mape_array(part[TARGET_COL].values, part["pred_power_w"].values),
        })

    return pd.DataFrame(rows).sort_values(["task_family", "model_name"]).reset_index(drop=True)


def train_from_fixed_params(train_df, val_df, model_tag, logger, fixed_params):
    params = BASE_XGB_PARAMS.copy()
    params.update(fixed_params)
    params["device"] = DEVICE

    X_train, y_train = df_to_xy(train_df)
    X_val, y_val = df_to_xy(val_df)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLS)

    logger.log(f"[INFO] Training with fixed params: {json.dumps(fixed_params, sort_keys=True)}")

    booster = train_booster_with_safe_mape(
        dtrain=dtrain,
        dval=dval,
        params=params,
        num_boost_round=MAX_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=VERBOSE_EVAL,
    )

    val_preds = predict_with_best_iteration(booster, X_val)
    best_record = {
        **fixed_params,
        "best_iteration": int(getattr(booster, "best_iteration", MAX_ESTIMATORS - 1)),
        "val_mape": float(safe_mape_array(y_val, val_preds)),
        "val_mae": float(mae_array(y_val, val_preds)),
        "val_rmse": float(rmse_array(y_val, val_preds)),
    }

    model_path = OUT_DIR / f"{model_tag}.json"
    best_json = META_DIR / f"{model_tag}_best_config.json"
    booster.save_model(model_path)
    save_json(best_json, best_record)

    return booster, best_record, model_path, None, best_json


# ============================================================
# Search strategy
# ============================================================

def build_search_grid():
    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


GLOBAL_GRID = build_search_grid()


def sample_search_grid(global_grid, n_trials, seed):
    if n_trials >= len(global_grid):
        return list(global_grid)
    rng = random.Random(seed)
    idx = list(range(len(global_grid)))
    rng.shuffle(idx)
    chosen = idx[:n_trials]
    return [global_grid[i] for i in chosen]


def run_config_search(train_df, val_df, model_tag, logger, candidate_configs):
    X_train, y_train = df_to_xy(train_df)
    X_val, y_val = df_to_xy(val_df)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLS)

    logger.log(f"[INFO] Device selected: {DEVICE}")
    logger.log(f"[INFO] Candidate count: {len(candidate_configs)}")

    best_record = None
    best_booster = None
    search_rows = []

    for idx, candidate in enumerate(candidate_configs, start=1):
        params = BASE_XGB_PARAMS.copy()
        params.update(candidate)
        params["device"] = DEVICE

        logger.log("")
        logger.log("==================================================")
        logger.log(f"[INFO] Candidate {idx}/{len(candidate_configs)}")
        logger.log(json.dumps(candidate, sort_keys=True))
        logger.log("==================================================")

        booster = train_booster_with_safe_mape(
            dtrain=dtrain,
            dval=dval,
            params=params,
            num_boost_round=MAX_ESTIMATORS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=VERBOSE_EVAL,
        )

        val_preds = predict_with_best_iteration(booster, X_val)
        val_mape = safe_mape_array(y_val, val_preds)
        val_mae = mae_array(y_val, val_preds)
        val_rmse = rmse_array(y_val, val_preds)

        best_iteration = int(getattr(booster, "best_iteration", MAX_ESTIMATORS - 1))

        record = {
            **candidate,
            "best_iteration": best_iteration,
            "val_mape": float(val_mape),
            "val_mae": float(val_mae),
            "val_rmse": float(val_rmse),
        }
        search_rows.append(record)

        logger.log(f"[INFO] best_iteration = {best_iteration}")
        logger.log(f"[INFO] val_mape      = {val_mape:.4f}%")
        logger.log(f"[INFO] val_mae       = {val_mae:.4f} W")
        logger.log(f"[INFO] val_rmse      = {val_rmse:.4f} W")

        if best_record is None or record["val_mape"] < best_record["val_mape"]:
            best_record = record
            best_booster = booster

    model_path = OUT_DIR / f"{model_tag}.json"
    search_csv = META_DIR / f"{model_tag}_search.csv"
    best_json = META_DIR / f"{model_tag}_best_config.json"

    best_booster.save_model(model_path)
    pd.DataFrame(search_rows).sort_values(["val_mape", "val_mae", "val_rmse"]).to_csv(search_csv, index=False)
    save_json(best_json, best_record)

    logger.log("")
    logger.log("[INFO] Best tuning result:")
    logger.log(json.dumps(best_record, indent=2, sort_keys=True))

    return best_booster, best_record, model_path, search_csv, best_json


def maybe_local_depth_candidates(base_params, logger=None):
    center = int(base_params["max_depth"])
    candidates = []

    for delta in LOCAL_DEPTH_NEIGHBORHOOD:
        d = center + delta
        if d < 2:
            continue

        if logger is not None and (
            d < min(SEARCH_SPACE["max_depth"]) or d > max(SEARCH_SPACE["max_depth"])
        ):
            logger.log(f"[INFO] Local depth candidate {d} is outside the global search range")

        p = dict(base_params)
        p["max_depth"] = d
        candidates.append(p)

    dedup = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            dedup.append(c)

    return dedup


def fit_model_with_policy(train_df, val_df, model_tag, logger, domain_best_params):
    if TUNING_POLICY == "reuse":
        return train_from_fixed_params(train_df, val_df, model_tag, logger, domain_best_params)

    local_candidates = maybe_local_depth_candidates(domain_best_params, logger=logger)
    logger.log(f"[INFO] Local adaptation candidate count = {len(local_candidates)}")
    return run_config_search(train_df, val_df, model_tag, logger, local_candidates)


# ============================================================
# Family-aware split
# ============================================================

def build_family_model_split(full_df, logger, min_train_models_per_family=1, min_test_models_per_family=1):
    rows = []

    for family, fam_df in full_df.groupby(TASK_FAMILY_COL):
        models = sorted(fam_df[MODEL_COL].unique().tolist())

        if len(models) < (min_train_models_per_family + min_test_models_per_family):
            logger.log(
                f"[WARN] Family={family}: only {len(models)} model(s); "
                f"cannot create train/test unseen-model split"
            )
            continue

        if len(models) == 2:
            n_test = 1
        elif len(models) == 3:
            n_test = 1
        else:
            n_test = max(1, round(0.3 * len(models)))

        rng = random.Random(deterministic_seed_from_text(f"family::{family}", RANDOM_STATE))
        shuffled = list(models)
        rng.shuffle(shuffled)

        test_models = sorted(shuffled[:n_test])
        train_models = sorted(shuffled[n_test:])

        rows.append({
            "task_family": family,
            "n_models": len(models),
            "train_models": train_models,
            "test_models": test_models,
        })

        logger.log(f"[INFO] Family={family}")
        logger.log(f"[INFO]   all models   = {models}")
        logger.log(f"[INFO]   train models = {train_models}")
        logger.log(f"[INFO]   test models  = {test_models}")

    return rows


def tune_once_per_domain_on_train_models(train_df, split_tag):
    rows = []
    best_params_by_domain = {}

    for domain in sorted(train_df[DOMAIN_COL].unique().tolist()):
        domain_df = train_df[train_df[DOMAIN_COL] == domain].copy().reset_index(drop=True)
        model_tag = f"{VERSION}_{split_tag}_domain_tune_{sanitize_name(domain)}"

        with DualLogger(LOG_DIR / f"{model_tag}.log") as logger:
            logger.log("============================================================")
            logger.log(f"[INFO] Domain-level tuning on TRAIN models only")
            logger.log(f"[INFO] Domain: {domain}")
            logger.log(f"[INFO] Split style: {CFG['split_desc']}")
            logger.log(f"[INFO] Search budget: {MAX_GLOBAL_SEARCH_TRIALS} sampled configs from {len(GLOBAL_GRID)} total")
            logger.log("============================================================")

            tr_df, val_df = internal_split_fn(
                domain_df,
                test_size=0.10,
                random_state=RANDOM_STATE,
                logger=logger,
                split_name=f"{split_tag} DOMAIN tune split ({domain})",
            )

            candidate_configs = sample_search_grid(
                GLOBAL_GRID,
                n_trials=MAX_GLOBAL_SEARCH_TRIALS,
                seed=deterministic_seed_from_text(f"{split_tag}::{domain}", RANDOM_STATE),
            )

            booster, best_record, model_path, search_csv, best_json = run_config_search(
                tr_df, val_df, model_tag, logger, candidate_configs
            )

            best_params_by_domain[domain] = {k: best_record[k] for k in SEARCH_SPACE.keys()}

            rows.append({
                "split_tag": split_tag,
                "domain": domain,
                "rows_total": len(domain_df),
                "best_iteration": best_record["best_iteration"],
                "best_val_mape_pct": best_record["val_mape"],
                "model_path": str(model_path),
                "search_csv": str(search_csv),
                "best_json": str(best_json),
                **{f"best_{k}": best_record[k] for k in SEARCH_SPACE.keys()},
            })

    summary_df = pd.DataFrame(rows).sort_values("domain").reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / f"{split_tag}_domain_tuning_summary.csv", index=False)
    save_json(META_DIR / f"{split_tag}_best_params_by_domain.json", best_params_by_domain)

    return best_params_by_domain, summary_df


# ============================================================
# Primary experiment: family-aware unseen-model generalization
# ============================================================

def run_family_generalization_experiment(full_df):
    split_tag = "family_generalization"

    with DualLogger(LOG_DIR / f"{split_tag}_bootstrap.log") as logger:
        logger.log("============================================================")
        logger.log("[INFO] Building family-aware unseen-model split")
        logger.log("============================================================")
        split_rows = build_family_model_split(full_df, logger)

    train_models = []
    test_models = []

    for row in split_rows:
        train_models.extend(row["train_models"])
        test_models.extend(row["test_models"])

    train_df = full_df[full_df[MODEL_COL].isin(train_models)].copy().reset_index(drop=True)
    test_df = full_df[full_df[MODEL_COL].isin(test_models)].copy().reset_index(drop=True)

    best_params_by_domain, tuning_summary = tune_once_per_domain_on_train_models(train_df, split_tag)

    rows = []

    for domain in sorted(train_df[DOMAIN_COL].unique().tolist()):
        domain_train = train_df[train_df[DOMAIN_COL] == domain].copy().reset_index(drop=True)
        domain_test = test_df[test_df[DOMAIN_COL] == domain].copy().reset_index(drop=True)

        model_tag = f"{VERSION}_{split_tag}_{sanitize_name(domain)}"

        with DualLogger(LOG_DIR / f"{model_tag}.log") as logger:
            logger.log("============================================================")
            logger.log("[INFO] Eyas-style family generalization experiment")
            logger.log(f"[INFO] Domain: {domain}")
            logger.log(f"[INFO] Train models: {sorted(domain_train[MODEL_COL].unique().tolist())}")
            logger.log(f"[INFO] Test models : {sorted(domain_test[MODEL_COL].unique().tolist())}")
            logger.log("============================================================")

            tr_df, val_df = internal_split_fn(
                domain_train,
                test_size=0.10,
                random_state=RANDOM_STATE,
                logger=logger,
                split_name=f"{split_tag} train/val split ({domain})",
            )

            booster, best_record, model_path, search_csv, best_json = fit_model_with_policy(
                tr_df,
                val_df,
                model_tag,
                logger,
                best_params_by_domain[domain],
            )

            train_metrics = evaluate_split(booster, tr_df, "Train", logger)
            val_metrics = evaluate_split(booster, val_df, "Val", logger)

            if len(domain_test) > 0:
                test_metrics = evaluate_split(booster, domain_test, "Test", logger)
                per_model_test = compute_per_model_mape(domain_test, test_metrics["preds"])
                per_model_csv = META_DIR / f"{model_tag}_per_model_test_mape.csv"
                per_model_test.to_csv(per_model_csv, index=False)
            else:
                logger.log(f"[WARN] No test rows available for domain={domain}")
                test_metrics = {"mape": np.nan, "mae": np.nan, "rmse": np.nan}
                per_model_csv = ""

            rows.append({
                "domain": domain,
                "train_models": ",".join(sorted(domain_train[MODEL_COL].unique().tolist())),
                "test_models": ",".join(sorted(domain_test[MODEL_COL].unique().tolist())),
                "train_families": ",".join(sorted(domain_train[TASK_FAMILY_COL].unique().tolist())),
                "test_families": ",".join(sorted(domain_test[TASK_FAMILY_COL].unique().tolist())),
                "train_rows": len(tr_df),
                "val_rows": len(val_df),
                "test_rows": len(domain_test),
                "best_iteration": best_record["best_iteration"],
                "best_val_mape_pct": best_record["val_mape"],
                "train_mape_pct": train_metrics["mape"],
                "val_mape_pct": val_metrics["mape"],
                "test_mape_pct": test_metrics["mape"],
                "model_path": str(model_path),
                "search_csv": str(search_csv) if search_csv is not None else "",
                "best_json": str(best_json),
                "per_model_test_csv": str(per_model_csv),
                **{f"best_{k}": best_record[k] for k in SEARCH_SPACE.keys() if k in best_record},
            })

    summary_df = pd.DataFrame(rows).sort_values("domain").reset_index(drop=True)
    summary_path = OUT_DIR / f"{split_tag}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    valid_test_count = int(summary_df["test_mape_pct"].notna().sum()) if len(summary_df) else 0
    avg_row = {c: "" for c in summary_df.columns}
    avg_row.update({
        "domain": "AVG",
        "train_rows": summary_df["train_rows"].mean() if len(summary_df) else np.nan,
        "val_rows": summary_df["val_rows"].mean() if len(summary_df) else np.nan,
        "test_rows": summary_df["test_rows"].mean() if len(summary_df) else np.nan,
        "best_iteration": summary_df["best_iteration"].mean() if len(summary_df) else np.nan,
        "best_val_mape_pct": summary_df["best_val_mape_pct"].mean() if len(summary_df) else np.nan,
        "train_mape_pct": summary_df["train_mape_pct"].mean() if len(summary_df) else np.nan,
        "val_mape_pct": summary_df["val_mape_pct"].mean() if len(summary_df) else np.nan,
        "test_mape_pct": summary_df["test_mape_pct"].mean() if len(summary_df) else np.nan,
        "train_models": f"valid_test_rows={valid_test_count}/{len(summary_df)}",
    })

    pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True).to_csv(
        OUT_DIR / f"{split_tag}_summary_with_avg.csv",
        index=False,
    )

    with DualLogger(LOG_DIR / f"{split_tag}_summary.log") as logger:
        logger.log(f"[INFO] Average test MAPE computed over {valid_test_count}/{len(summary_df)} valid domain rows")

    return summary_df, tuning_summary


# ============================================================
# Full deployable model
# ============================================================

def tune_once_per_domain_on_full_df(full_df, split_tag):
    rows = []
    best_params_by_domain = {}

    for domain in sorted(full_df[DOMAIN_COL].unique().tolist()):
        domain_df = full_df[full_df[DOMAIN_COL] == domain].copy().reset_index(drop=True)
        model_tag = f"{VERSION}_{split_tag}_domain_tune_{sanitize_name(domain)}"

        with DualLogger(LOG_DIR / f"{model_tag}.log") as logger:
            logger.log("============================================================")
            logger.log("[INFO] Domain-level tuning on FULL dataset")
            logger.log(f"[INFO] Domain: {domain}")
            logger.log("============================================================")

            tr_df, val_df = internal_split_fn(
                domain_df,
                test_size=0.10,
                random_state=RANDOM_STATE,
                logger=logger,
                split_name=f"{split_tag} DOMAIN tune split ({domain})",
            )

            candidate_configs = sample_search_grid(
                GLOBAL_GRID,
                n_trials=MAX_GLOBAL_SEARCH_TRIALS,
                seed=deterministic_seed_from_text(f"full::{domain}", RANDOM_STATE),
            )

            booster, best_record, model_path, search_csv, best_json = run_config_search(
                tr_df, val_df, model_tag, logger, candidate_configs
            )

            best_params_by_domain[domain] = {k: best_record[k] for k in SEARCH_SPACE.keys()}

            rows.append({
                "domain": domain,
                "best_iteration": best_record["best_iteration"],
                "best_val_mape_pct": best_record["val_mape"],
                "model_path": str(model_path),
                "search_csv": str(search_csv),
                "best_json": str(best_json),
                **{f"best_{k}": best_record[k] for k in SEARCH_SPACE.keys()},
            })

    summary_df = pd.DataFrame(rows).sort_values("domain").reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / f"{split_tag}_domain_tuning_summary.csv", index=False)
    save_json(META_DIR / f"{split_tag}_best_params_by_domain.json", best_params_by_domain)
    return best_params_by_domain, summary_df


def run_final_full_model(full_df):
    split_tag = "full_model"
    best_params_by_domain, tuning_summary = tune_once_per_domain_on_full_df(full_df, split_tag)

    rows = []

    for domain in sorted(full_df[DOMAIN_COL].unique().tolist()):
        domain_df = full_df[full_df[DOMAIN_COL] == domain].copy().reset_index(drop=True)
        model_tag = f"{VERSION}_{split_tag}_{sanitize_name(domain)}"

        with DualLogger(LOG_DIR / f"{model_tag}.log") as logger:
            logger.log("============================================================")
            logger.log("[INFO] Final full pooled model")
            logger.log(f"[INFO] Domain: {domain}")
            logger.log("============================================================")

            tr_df, val_df = internal_split_fn(
                domain_df,
                test_size=0.10,
                random_state=RANDOM_STATE,
                logger=logger,
                split_name=f"{split_tag} train/val split ({domain})",
            )

            booster, best_record, model_path, _, best_json = train_from_fixed_params(
                tr_df,
                val_df,
                model_tag,
                logger,
                best_params_by_domain[domain],
            )

            train_metrics = evaluate_split(booster, tr_df, "Train", logger)
            val_metrics = evaluate_split(booster, val_df, "Val", logger)

            rows.append({
                "domain": domain,
                "num_models": domain_df[MODEL_COL].nunique(),
                "train_rows": len(tr_df),
                "val_rows": len(val_df),
                "best_iteration": best_record["best_iteration"],
                "best_val_mape_pct": best_record["val_mape"],
                "train_mape_pct": train_metrics["mape"],
                "val_mape_pct": val_metrics["mape"],
                "model_path": str(model_path),
                "best_json": str(best_json),
                **{f"best_{k}": best_record[k] for k in SEARCH_SPACE.keys() if k in best_record},
            })

    summary_df = pd.DataFrame(rows).sort_values("domain").reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / f"{split_tag}_summary.csv", index=False)
    return summary_df, tuning_summary


# ============================================================
# Per-model baselines
# ============================================================

def run_separate_per_model_baselines(full_df, family_train_best_params_by_domain):
    rows = []

    for model_name in sorted(full_df[MODEL_COL].unique().tolist()):
        model_df = full_df[full_df[MODEL_COL] == model_name].copy().reset_index(drop=True)

        for domain in sorted(model_df[DOMAIN_COL].unique().tolist()):
            domain_model_df = model_df[model_df[DOMAIN_COL] == domain].copy().reset_index(drop=True)
            model_tag = f"{VERSION}_baseline_single_{sanitize_name(model_name)}__{sanitize_name(domain)}"

            with DualLogger(LOG_DIR / f"{model_tag}.log") as logger:
                logger.log("============================================================")
                logger.log(f"[INFO] Separate per-model baseline: {model_name}")
                logger.log(f"[INFO] Domain: {domain}")
                logger.log("============================================================")

                if len(domain_model_df) < MIN_ROWS_PER_MODEL_DOMAIN_BASELINE:
                    logger.log(
                        f"[WARN] Skipping {model_name}/{domain}: only {len(domain_model_df)} rows "
                        f"(below minimum threshold for stable 3-way split)"
                    )
                    continue

                temp_df, test_df = internal_split_fn(
                    domain_model_df,
                    test_size=TEST_SIZE,
                    random_state=RANDOM_STATE,
                    logger=logger,
                    split_name=f"{model_name} temp/test split ({domain})",
                )

                train_df, val_df = internal_split_fn(
                    temp_df,
                    test_size=VAL_SIZE_FROM_TEMP,
                    random_state=RANDOM_STATE,
                    logger=logger,
                    split_name=f"{model_name} train/val split ({domain})",
                )

                if domain not in family_train_best_params_by_domain:
                    logger.log(f"[WARN] No best params available for domain={domain}; skipping baseline")
                    continue

                booster, best_record, model_path, search_csv, best_json = fit_model_with_policy(
                    train_df,
                    val_df,
                    model_tag,
                    logger,
                    family_train_best_params_by_domain[domain],
                )

                train_metrics = evaluate_split(booster, train_df, "Train", logger)
                val_metrics = evaluate_split(booster, val_df, "Val", logger)
                test_metrics = evaluate_split(booster, test_df, "Test", logger)

                rows.append({
                    "model_name": model_name,
                    "task_family": domain_model_df[TASK_FAMILY_COL].iloc[0],
                    "domain": domain,
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                    "best_iteration": best_record["best_iteration"],
                    "best_val_mape_pct": best_record["val_mape"],
                    "train_mape_pct": train_metrics["mape"],
                    "val_mape_pct": val_metrics["mape"],
                    "test_mape_pct": test_metrics["mape"],
                    "model_path": str(model_path),
                    "search_csv": str(search_csv) if search_csv is not None else "",
                    "best_json": str(best_json),
                    **{f"best_{k}": best_record[k] for k in SEARCH_SPACE.keys() if k in best_record},
                })

    summary_df = pd.DataFrame(rows).sort_values(["task_family", "model_name", "domain"]).reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / "per_model_baseline_summary.csv", index=False)
    return summary_df


# ============================================================
# Optional auxiliary LOMO
# ============================================================

def run_aux_lomo(full_df, family_train_best_params_by_domain):
    rows = []
    held_out_models = sorted(full_df[MODEL_COL].unique().tolist())

    for fold_idx, held_out_model in enumerate(held_out_models, start=1):
        model_tag = f"{VERSION}_aux_lomo_holdout_{sanitize_name(held_out_model)}"

        with DualLogger(LOG_DIR / f"{model_tag}.log") as logger:
            logger.log("============================================================")
            logger.log("[INFO] Auxiliary LOMO robustness experiment")
            logger.log("[INFO] NOTE: this is stricter than Eyas and not the primary paper-faithful evaluation")
            logger.log(f"[INFO] Fold {fold_idx}/{len(held_out_models)}")
            logger.log(f"[INFO] Held-out model: {held_out_model}")
            logger.log("============================================================")

            test_df = full_df[full_df[MODEL_COL] == held_out_model].copy().reset_index(drop=True)
            train_pool_df = full_df[full_df[MODEL_COL] != held_out_model].copy().reset_index(drop=True)

            logger.log(
                "[NOTE] Domain-level hyperparameters were tuned on the family-train partition, "
                "not inside each LOMO fold. This is a deliberate cost-accuracy tradeoff."
            )

            for domain in sorted(train_pool_df[DOMAIN_COL].unique().tolist()):
                domain_train_pool = train_pool_df[train_pool_df[DOMAIN_COL] == domain].copy().reset_index(drop=True)
                domain_test = test_df[test_df[DOMAIN_COL] == domain].copy().reset_index(drop=True)

                if len(domain_train_pool) == 0 or domain not in family_train_best_params_by_domain:
                    continue

                train_df, val_df = internal_split_fn(
                    domain_train_pool,
                    test_size=0.10,
                    random_state=RANDOM_STATE,
                    logger=logger,
                    split_name=f"aux LOMO train/val split ({domain})",
                )

                booster, best_record, model_path, search_csv, best_json = fit_model_with_policy(
                    train_df,
                    val_df,
                    f"{model_tag}__{sanitize_name(domain)}",
                    logger,
                    family_train_best_params_by_domain[domain],
                )

                train_metrics = evaluate_split(booster, train_df, "Train", logger)
                val_metrics = evaluate_split(booster, val_df, "Val", logger)

                if len(domain_test) > 0:
                    test_metrics = evaluate_split(booster, domain_test, "Test", logger)
                    test_mape = test_metrics["mape"]
                else:
                    logger.log(f"[WARN] No held-out test rows for domain={domain}")
                    test_mape = np.nan

                rows.append({
                    "held_out_model": held_out_model,
                    "domain": domain,
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(domain_test),
                    "best_iteration": best_record["best_iteration"],
                    "best_val_mape_pct": best_record["val_mape"],
                    "train_mape_pct": train_metrics["mape"],
                    "val_mape_pct": val_metrics["mape"],
                    "test_mape_pct": test_mape,
                    "model_path": str(model_path),
                    "search_csv": str(search_csv) if search_csv is not None else "",
                    "best_json": str(best_json),
                })

    summary_df = pd.DataFrame(rows).sort_values(["held_out_model", "domain"]).reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / "aux_lomo_summary.csv", index=False)

    valid_test_count = int(summary_df["test_mape_pct"].notna().sum()) if len(summary_df) else 0
    with DualLogger(LOG_DIR / "aux_lomo_summary.log") as logger:
        logger.log(f"[INFO] Average test MAPE computed over {valid_test_count}/{len(summary_df)} valid rows")

    return summary_df


# ============================================================
# Main
# ============================================================

def main():
    with DualLogger(LOG_DIR / "bootstrap.log") as logger:
        logger.log(f"[INFO] Bootstrapping Version {VERSION}")
        logger.log(f"[INFO] Tuning policy: {TUNING_POLICY}")
        logger.log(f"[INFO] Device selected: {DEVICE}")
        logger.log(f"[INFO] Dataset directory: {DATASET_DIR}")
        logger.log(f"[INFO] Internal split mode: {CFG['internal_split_mode']}")
        logger.log(f"[INFO] Full grid size: {len(GLOBAL_GRID)}")
        logger.log(f"[INFO] Search budget per tuning run: {MAX_GLOBAL_SEARCH_TRIALS}")
        logger.log(f"[INFO] Automatically discovered {len(CFG['csv_files'])} dataset files:")
        for model_name, csv_path in sorted(CFG["csv_files"].items()):
            logger.log(f"    - {model_name}: {csv_path}")

        full_df = load_all_datasets(CFG["csv_files"], logger)
        validate_family_inventory(full_df, logger)

    family_summary, family_tuning_summary = run_family_generalization_experiment(full_df)
    full_summary, full_tuning_summary = run_final_full_model(full_df)

    family_best_params_path = META_DIR / "family_generalization_best_params_by_domain.json"
    with open(family_best_params_path, "r", encoding="utf-8") as fp:
        family_train_best_params_by_domain = json.load(fp)

    baseline_summary = run_separate_per_model_baselines(full_df, family_train_best_params_by_domain)

    if RUN_AUX_L0MO:
        aux_lomo_summary = run_aux_lomo(full_df, family_train_best_params_by_domain)
    else:
        aux_lomo_summary = pd.DataFrame()

    print("\n============================================================")
    print(f"[INFO] VERSION {VERSION} COMPLETED")
    print("[INFO] Family generalization summary:")
    print(family_summary.round(4).to_string(index=False))
    print("[INFO] Full model summary:")
    print(full_summary.round(4).to_string(index=False))
    print("[INFO] Per-model baseline summary:")
    print(baseline_summary.round(4).to_string(index=False))
    if len(aux_lomo_summary):
        print("[INFO] Auxiliary LOMO summary:")
        print(aux_lomo_summary.round(4).to_string(index=False))
    print("============================================================\n")


if __name__ == "__main__":
    main()