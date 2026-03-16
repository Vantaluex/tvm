import csv
import time
import queue
import hashlib
import multiprocessing as mp

import numpy as np
import tvm
from tvm import tir
from tvm import meta_schedule as ms


# =========================
# USER SETTINGS
# =========================
MODELNAME = "deepseekr1-qwen-14b"
DB_WORK_DIR = "complete_tuning_logs_20000/tuning_logs_" + MODELNAME

# Pick a small slice first.
# Example: records you already tested successfully around 3998.
TEST_RECORDS = [30, 31, 32, 33, 3995, 3996, 3997, 3998]

# Compare these settings against each other.
MIN_REPEAT_MS_LIST = [3000, 2000, 1500, 1000, 500, 300, 200, 100]

# Keep these fixed while varying min_repeat_ms.
NUMBER = 1
REPEAT = 1

# Per-measurement safety timeout.
TIMEOUT_SECONDS = 90

# Output file
OUT_CSV = "min_repeat_ms_sweep.csv"


def as_float_metric(x):
    if hasattr(x, "ratio"):
        return float(x.ratio)
    return float(x)


def trace_fingerprint(trace):
    obj = trace.as_python() if hasattr(trace, "as_python") else repr(trace)
    if isinstance(obj, str):
        s = obj
    else:
        s = "\n".join(str(x) for x in obj)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


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
        x = np.random.rand(*shape).astype("float32")
        x = x.astype(dtype)

    return tvm.nd.array(x, device=dev)


def measure_one(abs_idx, db_work_dir, min_repeat_ms, result_queue):
    try:
        t0 = time.time()

        dev = tvm.cuda(0)
        db = ms.database.JSONDatabase(work_dir=db_work_dir)
        all_recs = db.get_all_tuning_records()
        r = all_recs[abs_idx]

        mod = r.workload.mod
        target = r.target
        workload_hash = int(tvm.ir.structural_hash(mod))
        trace_hash = trace_fingerprint(r.trace)

        sch = tir.Schedule(mod, debug_mask="all")
        r.trace.apply_to_schedule(sch, remove_postproc=True)

        cand = ms.MeasureCandidate(sch=sch, args_info=r.args_info)
        ctx = ms.TuneContext(mod=mod, target=target, task_name=f"rec_{abs_idx}")

        extractor = ms.feature_extractor.PerStoreFeature()
        (feat_nd,) = extractor.extract_from(ctx, candidates=[cand])
        feat = feat_nd.numpy()
        n_stores = int(feat.shape[0])

        rt_mod = tvm.build(sch.mod, target=target)
        args = [make_nd(t.shape, t.dtype, dev) for t in r.args_info]

        ftimer = rt_mod.time_evaluator(
            "main",
            dev,
            number=NUMBER,
            repeat=REPEAT,
            min_repeat_ms=min_repeat_ms,
        )

        t1 = time.time()
        timing = ftimer(*args)
        t2 = time.time()

        nvml = tvm.get_global_func("runtime.profiling.get_last_nvml_metrics")()
        avg_power = as_float_metric(nvml["avg_power_w"])

        result_queue.put(
            (
                "ok",
                {
                    "record": abs_idx,
                    "workload_hash": workload_hash,
                    "trace_hash": trace_hash,
                    "n_stores": n_stores,
                    "min_repeat_ms": min_repeat_ms,
                    "lat_mean_ms": float(timing.mean) * 1e3,
                    "avg_power_w": float(avg_power),
                    "setup_plus_build_s": t1 - t0,
                    "ftimer_call_s": t2 - t1,
                    "total_worker_s": t2 - t0,
                },
            )
        )

    except Exception as e:
        result_queue.put(
            (
                "err",
                {
                    "record": abs_idx,
                    "min_repeat_ms": min_repeat_ms,
                    "error": str(e),
                },
            )
        )


def run_one_with_timeout(ctx, abs_idx, min_repeat_ms):
    result_queue = ctx.Queue()
    p = ctx.Process(
        target=measure_one,
        args=(abs_idx, DB_WORK_DIR, min_repeat_ms, result_queue),
    )
    p.start()
    p.join(TIMEOUT_SECONDS)

    if p.is_alive():
        p.terminate()
        p.join()
        result_queue.close()
        result_queue.join_thread()
        return {
            "record": abs_idx,
            "min_repeat_ms": min_repeat_ms,
            "status": "timeout",
            "workload_hash": "",
            "trace_hash": "",
            "n_stores": "",
            "lat_mean_ms": "",
            "avg_power_w": "",
            "setup_plus_build_s": "",
            "ftimer_call_s": "",
            "total_worker_s": "",
            "error": f"timeout after {TIMEOUT_SECONDS}s",
        }

    try:
        status, payload = result_queue.get(timeout=5)
    except queue.Empty:
        result_queue.close()
        result_queue.join_thread()
        return {
            "record": abs_idx,
            "min_repeat_ms": min_repeat_ms,
            "status": "no_result",
            "workload_hash": "",
            "trace_hash": "",
            "n_stores": "",
            "lat_mean_ms": "",
            "avg_power_w": "",
            "setup_plus_build_s": "",
            "ftimer_call_s": "",
            "total_worker_s": "",
            "error": "queue empty",
        }

    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        payload["status"] = "ok"
        payload["error"] = ""
        return payload

    return {
        "record": payload["record"],
        "min_repeat_ms": payload["min_repeat_ms"],
        "status": "error",
        "workload_hash": "",
        "trace_hash": "",
        "n_stores": "",
        "lat_mean_ms": "",
        "avg_power_w": "",
        "setup_plus_build_s": "",
        "ftimer_call_s": "",
        "total_worker_s": "",
        "error": payload["error"],
    }


def summarize(rows):
    ok_rows = [r for r in rows if r["status"] == "ok"]
    by_record = {}
    for r in ok_rows:
        by_record.setdefault(r["record"], {})[r["min_repeat_ms"]] = r

    baseline_ms = max(MIN_REPEAT_MS_LIST)
    print("\n=== Relative change vs baseline ===")
    print(f"Baseline min_repeat_ms = {baseline_ms}")

    deltas = []
    for rec in sorted(by_record):
        if baseline_ms not in by_record[rec]:
            continue
        base = by_record[rec][baseline_ms]
        base_p = base["avg_power_w"]
        base_l = base["lat_mean_ms"]

        for ms_value in MIN_REPEAT_MS_LIST:
            if ms_value == baseline_ms or ms_value not in by_record[rec]:
                continue
            cur = by_record[rec][ms_value]

            power_delta_pct = 100.0 * (cur["avg_power_w"] - base_p) / base_p if base_p != 0 else np.nan
            lat_delta_pct = 100.0 * (cur["lat_mean_ms"] - base_l) / base_l if base_l != 0 else np.nan
            deltas.append((rec, ms_value, power_delta_pct, lat_delta_pct))

            print(
                f"record {rec:5d} | {ms_value:4d} ms | "
                f"power delta = {power_delta_pct:+7.3f}% | "
                f"lat delta = {lat_delta_pct:+7.3f}%"
            )

    if deltas:
        print("\n=== Aggregate absolute deltas vs baseline ===")
        for ms_value in MIN_REPEAT_MS_LIST:
            if ms_value == baseline_ms:
                continue
            sub = [d for d in deltas if d[1] == ms_value]
            if not sub:
                continue
            mean_abs_power = float(np.mean([abs(x[2]) for x in sub]))
            max_abs_power = float(np.max([abs(x[2]) for x in sub]))
            mean_abs_lat = float(np.mean([abs(x[3]) for x in sub]))
            print(
                f"{ms_value:4d} ms | mean |power delta| = {mean_abs_power:7.3f}% | "
                f"max |power delta| = {max_abs_power:7.3f}% | "
                f"mean |lat delta| = {mean_abs_lat:7.3f}%"
            )


def main():
    ctx = mp.get_context("spawn")
    rows = []

    print("[INFO] Starting min_repeat_ms sweep")
    print(f"[INFO] Records: {TEST_RECORDS}")
    print(f"[INFO] min_repeat_ms values: {MIN_REPEAT_MS_LIST}")
    print(f"[INFO] Timeout: {TIMEOUT_SECONDS}s")

    for abs_idx in TEST_RECORDS:
        print(f"\n[INFO] Record {abs_idx}")
        for min_repeat_ms in MIN_REPEAT_MS_LIST:
            row = run_one_with_timeout(ctx, abs_idx, min_repeat_ms)
            rows.append(row)

            if row["status"] == "ok":
                print(
                    f"  min_repeat_ms={min_repeat_ms:4d} | "
                    f"lat={row['lat_mean_ms']:.3f} ms | "
                    f"power={row['avg_power_w']:.3f} W | "
                    f"ftimer_call={row['ftimer_call_s']:.3f} s"
                )
            else:
                print(
                    f"  min_repeat_ms={min_repeat_ms:4d} | "
                    f"status={row['status']} | {row['error']}"
                )

    fieldnames = [
        "record",
        "min_repeat_ms",
        "status",
        "workload_hash",
        "trace_hash",
        "n_stores",
        "lat_mean_ms",
        "avg_power_w",
        "setup_plus_build_s",
        "ftimer_call_s",
        "total_worker_s",
        "error",
    ]

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\n[INFO] Wrote {OUT_CSV}")
    summarize(rows)


if __name__ == "__main__":
    main()
