import os
import csv
import time
import random
import hashlib
import threading

import numpy as np
import pynvml
import tvm
from tvm import tir
from tvm import meta_schedule as ms
from collections import defaultdict



# ============================================================
# CONFIG: EDIT ONLY THIS BLOCK
# ============================================================

CONFIG = {
    "root_dir": "complete_tuning_logs_20000",
    "out_csv": "power_selected_records.csv",
    "fail_csv": "power_selected_records_failures.csv",
    "gpu_index": 0,
    "nvml_sample_ms": 20,
    "random_seed": 20260325,
    "error_csv": "power_selected_records_errors.csv",
    "error_summary_csv": "power_selected_records_error_summary_per_model.csv",
}

MEASUREMENT_SCENARIOS = [
    {"name": "baseline", "warmup_s": 2.5, "measure_s": 0.5},
    {"name": "fast", "warmup_s": 1.2, "measure_s": 0.3},
    {"name": "faster", "warmup_s": 1, "measure_s": 0.3},
]

# RUN_PLAN is executed top-to-bottom in exactly this order.
# For each entry:
# - use "records": [...] for exact record indices in exact order
# - use "random": N for N random records from that model
#
# Examples:
# {"model": "densenet169", "records": [91, 7, 120, 3, 55]}
# {"model": "resnet50", "random": 5}

RUN_PLAN = [
    {"model": "convnextv2-tiny", "random": 5},
    {"model": "deepseekr1-qwen-14b", "random": 5},
    {"model": "exaone-deep-7.8B", "random": 5},
    {"model": "llama-3.1-8b", "random": 5},
    {"model": "mobilenetv3large", "random": 5},
    {"model": "qwen2.5-3b", "random": 5},
    {"model": "resnet50", "random": 5},
    {"model": "segformer-b2", "random": 5},
    {"model": "deberta-v3-base", "random": 5},
    {"model": "densenet169", "random": 5},
    {"model": "exaone3.5-7.8b", "random": 5},
    {"model": "mask2former-swin-small", "random": 5},
    {"model": "modernbert-base", "random": 5},
    {"model": "qwen2.5-9B", "random": 5},
    {"model": "rtdetr-r50", "random": 5},
]

# Example custom manual order:
# RUN_PLAN = [
#     {"model": "densenet169", "records": [88, 3, 401, 20, 1]},
#     {"model": "resnet50", "records": [7, 99, 2]},
#     {"model": "segformer-b2", "random": 5},
#     {"model": "densenet169", "records": [55, 12]},  # allowed, runs later in this exact spot
# ]


# ============================================================
# HELPERS
# ============================================================
def scenario_total_seconds(s):
    return s["warmup_s"] + s["measure_s"]


def scenario_windows(s):
    w = s["warmup_s"]
    t = scenario_total_seconds(s)
    return [
        ("avg_power_0_to_warmup_w", 0.0, w),
        ("avg_power_warmup_to_total_w", w, t),
        ("avg_power_0_to_total_w", 0.0, t),
    ]


def make_nd(shape, dtype, dev):
    shape = [int(s) for s in shape]
    dtype = str(dtype)

    if "float" in dtype:
        x = np.random.rand(*shape).astype(dtype)
    elif "int" in dtype:
        info = np.iinfo(np.dtype(dtype))
        lo = max(info.min, -8)
        hi = min(info.max, 8)
        if lo >= hi:
            lo, hi = 0, 1
        x = np.random.randint(lo, hi + 1, size=shape, dtype=np.dtype(dtype))
    elif dtype == "bool":
        x = np.random.randint(0, 2, size=shape).astype("bool")
    else:
        x = np.random.rand(*shape).astype("float32").astype(dtype)

    return tvm.nd.array(x, device=dev)


def trace_fingerprint(trace):
    obj = trace.as_python() if hasattr(trace, "as_python") else repr(trace)
    if isinstance(obj, str):
        s = obj
    else:
        s = "\n".join(str(x) for x in obj)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def model_dir(model_name):
    return os.path.join(CONFIG["root_dir"], f"tuning_logs_{model_name}")


def get_device_for_record(record):
    kind = str(record.target.kind.name)
    if kind != "cuda":
        raise RuntimeError(f"Expected CUDA target, got {kind}")
    return tvm.cuda(CONFIG["gpu_index"])


def poll_nvml_power(stop_event, samples, t0):
    handle = pynvml.nvmlDeviceGetHandleByIndex(CONFIG["gpu_index"])
    interval_s = CONFIG["nvml_sample_ms"] / 1000.0

    while not stop_event.is_set():
        now = time.perf_counter() - t0
        power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        samples.append((now, power_w))
        time.sleep(interval_s)


def avg_power_over_window(samples, start_s, end_s):
    if end_s <= start_s:
        raise ValueError(f"Invalid window: {start_s} -> {end_s}")
    if not samples:
        return float("nan")

    samples = sorted(samples, key=lambda x: x[0])

    current_p = samples[0][1]
    for t, p in samples:
        if t <= start_s:
            current_p = p
        else:
            break

    energy = 0.0
    prev_t = start_s

    for t, p in samples:
        if t <= start_s:
            continue

        seg_end = min(t, end_s)
        if seg_end > prev_t:
            energy += current_p * (seg_end - prev_t)
            prev_t = seg_end

        if t >= end_s:
            break

        current_p = p

    if prev_t < end_s:
        energy += current_p * (end_s - prev_t)

    return energy / (end_s - start_s)


def run_main_with_trace(rt_mod, args, dev, scenario):
    f = rt_mod["main"]

    samples = []
    iter_ms = []
    stop_event = threading.Event()

    t0 = time.perf_counter()
    poller = threading.Thread(
        target=poll_nvml_power,
        args=(stop_event, samples, t0),
        daemon=True,
    )
    poller.start()

    try:
        while True:
            now = time.perf_counter()
            if now - t0 >= scenario_total_seconds(scenario):
                break

            t1 = time.perf_counter()
            f(*args)
            dev.sync()
            t2 = time.perf_counter()
            iter_ms.append((t2 - t1) * 1000.0)
    finally:
        stop_event.set()
        poller.join()

    elapsed_s = time.perf_counter() - t0
    if samples:
        last_t, last_p = samples[-1]
        if last_t < elapsed_s:
            samples.append((elapsed_s, last_p))

    return {
        "elapsed_s": elapsed_s,
        "iters": len(iter_ms),
        "avg_iter_ms": float(np.mean(iter_ms)) if iter_ms else float("nan"),
        "samples": samples,
    }



def extract_features_and_build(record, run_tag):
    dev = get_device_for_record(record)

    mod = record.workload.mod
    sch = tir.Schedule(mod, debug_mask="all")
    record.trace.apply_to_schedule(sch, remove_postproc=True)

    cand = ms.MeasureCandidate(sch=sch, args_info=record.args_info)
    ctx = ms.TuneContext(mod=mod, target=record.target, task_name=run_tag)

    extractor = ms.feature_extractor.PerStoreFeature()
    (feat_nd,) = extractor.extract_from(ctx, candidates=[cand])
    feat = feat_nd.numpy()

    if feat.ndim != 2 or feat.shape[0] == 0:
        raise RuntimeError(f"Unexpected feature shape: {feat.shape}")

    agg = np.concatenate([
        feat.mean(0),
        feat.std(0),
        feat.min(0),
        feat.max(0),
    ])

    rt_mod = tvm.build(sch.mod, target=record.target)
    args = [make_nd(t.shape, t.dtype, dev) for t in record.args_info]

    return dev, feat, agg, rt_mod, args


def measure_record_under_scenario(model_name, abs_idx, record, scenario):
    dev, feat, agg, rt_mod, args = extract_features_and_build(
        record, run_tag=f"{model_name}_{abs_idx}_{scenario['name']}"
    )

    trace = run_main_with_trace(rt_mod, args, dev, scenario)

    power_values = {}
    for col, start_s, end_s in scenario_windows(scenario):
        power_values[col] = avg_power_over_window(trace["samples"], start_s, end_s)

    meta = {
        "model": model_name,
        "record_index": abs_idx,
        "scenario": scenario["name"],
        "warmup_s": float(scenario["warmup_s"]),
        "measure_s": float(scenario["measure_s"]),
        "total_s": float(scenario_total_seconds(scenario)),
        "workload_hash": int(tvm.ir.structural_hash(record.workload.mod)),
        "trace_hash": trace_fingerprint(record.trace),
        "n_stores": int(feat.shape[0]),
        "elapsed_s": float(trace["elapsed_s"]),
        "iters": int(trace["iters"]),
        "avg_iter_ms": float(trace["avg_iter_ms"]),
    }

    return meta, power_values, agg



def load_records_for_model(model_name):
    db_path = model_dir(model_name)
    if not os.path.isdir(db_path):
        raise RuntimeError(f"Missing directory: {db_path}")

    db = ms.database.JSONDatabase(work_dir=db_path)
    return db.get_all_tuning_records()


def choose_record_indices(entry, n_records, rng):
    if "records" in entry and "random" in entry:
        raise RuntimeError(f"Use either 'records' or 'random', not both: {entry}")

    if "records" in entry:
        indices = entry["records"]
        if not isinstance(indices, list):
            raise RuntimeError(f"'records' must be a list: {entry}")
        for idx in indices:
            if idx < 0 or idx >= n_records:
                raise RuntimeError(f"Record index out of range: {idx} / {n_records}")
        return indices

    if "random" in entry:
        k = int(entry["random"])
        all_idx = list(range(n_records))
        rng.shuffle(all_idx)
        return all_idx[: min(k, n_records)]

    raise RuntimeError(f"Entry must contain either 'records' or 'random': {entry}")


def build_header(agg_dim):
    return [
        "model",
        "record_index",
        "scenario",
        "warmup_s",
        "measure_s",
        "total_s",
        "workload_hash",
        "trace_hash",
        "n_stores",
        "avg_power_0_to_warmup_w",
        "avg_power_warmup_to_total_w",
        "avg_power_0_to_total_w",
        "elapsed_s",
        "iters",
        "avg_iter_ms",
        *[f"f{k}" for k in range(agg_dim)],
    ]


def write_baseline_error_csv():
    in_csv = CONFIG["out_csv"]
    out_csv = CONFIG["error_csv"]

    with open(in_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[WARN] No rows found in raw output CSV, skipping error CSV.")
        return

    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["record_index"])].append(row)

    result_rows = []

    for (model, record_index), group_rows in groups.items():
        baseline_row = max(group_rows, key=lambda r: float(r["warmup_s"]))

        baseline_scenario = baseline_row["scenario"]
        baseline_warmup_s = float(baseline_row["warmup_s"])
        baseline_measure_s = float(baseline_row["measure_s"])
        baseline_total_s = float(baseline_row["total_s"])

        baseline_power = float(baseline_row["avg_power_warmup_to_total_w"])

        for row in group_rows:
            test_power = float(row["avg_power_warmup_to_total_w"])
            signed_error_w = test_power - baseline_power
            absolute_error_w = abs(signed_error_w)

            if baseline_power == 0:
                signed_error_pct = float("nan")
                absolute_error_pct = float("nan")
            else:
                signed_error_pct = 100.0 * signed_error_w / baseline_power
                absolute_error_pct = abs(signed_error_pct)

            result_rows.append([
                model,
                record_index,
                row["scenario"],
                float(row["warmup_s"]),
                float(row["measure_s"]),
                float(row["total_s"]),
                baseline_scenario,
                baseline_warmup_s,
                baseline_measure_s,
                baseline_total_s,
                baseline_power,
                test_power,
                signed_error_w,
                absolute_error_w,
                signed_error_pct,
                absolute_error_pct,
            ])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model",
            "record_index",
            "scenario",
            "warmup_s",
            "measure_s",
            "total_s",
            "baseline_scenario",
            "baseline_warmup_s",
            "baseline_measure_s",
            "baseline_total_s",
            "baseline_power_warmup_to_total_w",
            "test_power_warmup_to_total_w",
            "signed_error_w",
            "absolute_error_w",
            "signed_error_pct",
            "absolute_error_pct",
        ])
        w.writerows(result_rows)

    print("Wrote:", out_csv)


def write_baseline_error_summary_per_model():
    in_csv = CONFIG["error_csv"]
    out_csv = CONFIG["error_summary_csv"]

    if not os.path.exists(in_csv):
        print(f"[WARN] Missing error CSV: {in_csv}")
        return

    with open(in_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[WARN] No rows found in error CSV, skipping summary.")
        return

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["scenario"])].append(row)

    summary_rows = []

    for (model, scenario), group_rows in sorted(grouped.items()):
        abs_err_w = [float(r["absolute_error_w"]) for r in group_rows]
        abs_err_pct = [float(r["absolute_error_pct"]) for r in group_rows]
        signed_err_w = [float(r["signed_error_w"]) for r in group_rows]
        signed_err_pct = [float(r["signed_error_pct"]) for r in group_rows]

        summary_rows.append([
            model,
            scenario,
            len(group_rows),
            float(np.mean(abs_err_w)),
            float(np.max(abs_err_w)),
            float(np.mean(abs_err_pct)),
            float(np.max(abs_err_pct)),
            float(np.mean(signed_err_w)),
            float(np.mean(signed_err_pct)),
        ])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model",
            "scenario",
            "n_rows",
            "mae_w",
            "max_ae_w",
            "mape_pct",
            "max_ape_pct",
            "mean_signed_error_w",
            "mean_signed_error_pct",
        ])
        w.writerows(summary_rows)

    print("Wrote:", out_csv)





# ============================================================
# MAIN
# ============================================================

def main():
    rng = random.Random(CONFIG["random_seed"])
    pynvml.nvmlInit()

    header_written = False
    agg_dim = None

    try:
        with open(CONFIG["out_csv"], "w", newline="") as out_f, open(CONFIG["fail_csv"], "w", newline="") as fail_f:
            out_w = csv.writer(out_f)
            fail_w = csv.writer(fail_f)

            fail_w.writerow(["model", "record_index", "scenario", "reason"])
            fail_f.flush()

            for plan_pos, entry in enumerate(RUN_PLAN):
                model_name = entry["model"]
                print(f"\n\n[MODEL {plan_pos}] model={model_name}")

                try:
                    records = load_records_for_model(model_name)
                    selected_indices = choose_record_indices(entry, len(records), rng)
                    print(f"[INFO] selected_indices={selected_indices}")
                except Exception as e:
                    fail_w.writerow([model_name, "", "", f"setup_error: {e}"])
                    fail_f.flush()
                    print(f"[FAIL] setup {model_name} -> {e}")
                    continue

                for abs_idx in selected_indices:
                    print(f"\n[RUN] {model_name} idx={abs_idx}")

                    for scenario in MEASUREMENT_SCENARIOS:
                        scenario_name = scenario["name"]
                        print(
                            f"[SCENARIO] {scenario_name} "
                            f"(warmup={scenario['warmup_s']}s, measure={scenario['measure_s']}s)"
                        )

                        try:
                            meta, power_values, agg = measure_record_under_scenario(
                                model_name=model_name,
                                abs_idx=abs_idx,
                                record=records[abs_idx],
                                scenario=scenario,
                            )

                            if not header_written:
                                agg_dim = len(agg)
                                out_w.writerow(build_header(agg_dim))
                                out_f.flush()
                                header_written = True
                            elif len(agg) != agg_dim:
                                raise RuntimeError(
                                    f"Feature dimension mismatch: got {len(agg)}, expected {agg_dim}"
                                )

                            row = [
                                meta["model"],
                                meta["record_index"],
                                meta["scenario"],
                                meta["warmup_s"],
                                meta["measure_s"],
                                meta["total_s"],
                                meta["workload_hash"],
                                meta["trace_hash"],
                                meta["n_stores"],
                                power_values["avg_power_0_to_warmup_w"],
                                power_values["avg_power_warmup_to_total_w"],
                                power_values["avg_power_0_to_total_w"],
                                meta["elapsed_s"],
                                meta["iters"],
                                meta["avg_iter_ms"],
                                *[float(x) for x in agg.tolist()],
                            ]

                            out_w.writerow(row)
                            out_f.flush()

                            print(
                                f"[OK] {model_name} idx={abs_idx} scenario={scenario_name} | "
                                f"0~{meta['warmup_s']}={power_values['avg_power_0_to_warmup_w']:.2f}W | "
                                f"{meta['warmup_s']}~{meta['total_s']}={power_values['avg_power_warmup_to_total_w']:.2f}W | "
                                f"0~{meta['total_s']}={power_values['avg_power_0_to_total_w']:.2f}W"
                            )

                        except Exception as e:
                            fail_w.writerow([model_name, abs_idx, scenario_name, str(e)])
                            fail_f.flush()
                            print(f"[FAIL] {model_name} idx={abs_idx} scenario={scenario_name} -> {e}")

        print("\nWrote:", CONFIG["out_csv"])
        print("Wrote:", CONFIG["fail_csv"])
        write_baseline_error_csv()
        write_baseline_error_summary_per_model()

    finally:
        pynvml.nvmlShutdown()



if __name__ == "__main__":
    main()
