import csv
import numpy as np
import tvm
from tvm import tir
from tvm import meta_schedule as ms
import hashlib
import subprocess
import atexit
import multiprocessing as mp
import queue


# # === CLOCK LOCKING FOR WSL -> WINDOWS ===
# def lock_clocks_windows(freq=2618):
#     print(f"\n[INFO] Attempting to lock Windows GPU clocks to {freq} MHz...")
#     cmd = f"Start-Process nvidia-smi.exe -ArgumentList '-i 0 -lgc {freq},{freq}' -Verb RunAs -Wait"
#     try:
#         subprocess.run(["powershell.exe", "-Command", cmd], check=True)
#         print("[INFO] GPU clocks locked successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"[WARN] Failed to lock clocks. Ensure you accepted the UAC prompt. Error: {e}")


# def unlock_clocks_windows():
#     print(f"\n[INFO] Attempting to unlock Windows GPU clocks...")
#     cmd = "Start-Process nvidia-smi.exe -ArgumentList '-i 0 -rgc' -Verb RunAs -Wait"
#     try:
#         subprocess.run(["powershell.exe", "-Command", cmd], check=True)
#         print("[INFO] GPU clocks unlocked successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"[WARN] Failed to unlock clocks. Error: {e}")


# # Ensure clocks unlock even if script crashes
# atexit.register(unlock_clocks_windows)
# # ========================================


def as_float_metric(x):
    if hasattr(x, "ratio"):
        return float(x.ratio)
    return float(x)


def make_nd(shape, dtype, dev):
    x = np.random.rand(*[int(s) for s in shape]).astype(dtype)
    return tvm.nd.array(x, device=dev)


def trace_fingerprint(trace):
    obj = trace.as_python() if hasattr(trace, "as_python") else repr(trace)
    if isinstance(obj, str):
        s = obj
    else:
        s = "\n".join(str(x) for x in obj)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# HYPERPARAMETERS
N = 20000
start_N = 4000
MODELNAME = "deepseekr1-qwen-14b"
FREQ = "2295"
TIMEOUT_SECONDS = 60
# FREQ = 2618 # Base clock lock freq


def process_one_record(abs_idx, db_work_dir, result_queue):
    try:
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

        agg = np.concatenate([feat.mean(0), feat.std(0), feat.min(0), feat.max(0)])

        rt_mod = tvm.build(sch.mod, target=target)
        args = [make_nd(t.shape, t.dtype, dev) for t in r.args_info]

        ftimer = rt_mod.time_evaluator("main", dev, number=1, repeat=1, min_repeat_ms=1000)
        timing = ftimer(*args)

        nvml = tvm.get_global_func("runtime.profiling.get_last_nvml_metrics")()
        avg_power = as_float_metric(nvml["avg_power_w"])

        row = [
            abs_idx,
            workload_hash,
            trace_hash,
            int(feat.shape[0]),
            float(timing.mean) * 1e3,
            float(avg_power),
        ]
        row += [float(x) for x in agg.tolist()]

        result_queue.put(
            (
                "ok",
                row,
                workload_hash,
                float(timing.mean) * 1e3,
                float(avg_power),
            )
        )

    except Exception as e:
        result_queue.put(("err", abs_idx, str(e)))


def main():
    # # Lock clocks before starting measurements
    # lock_clocks_windows(freq=FREQ)

    db_work_dir = "complete_tuning_logs_20000/tuning_logs_" + MODELNAME
    db = ms.database.JSONDatabase(work_dir=db_work_dir)
    all_recs = db.get_all_tuning_records()

    actual_N = min(N, len(all_recs))
    recs = [i for i in range(start_N, actual_N)]
    print(f"[INFO] Loaded {len(recs)} tuning records from database.")

    strstart_N = str(start_N)
    out_path = "dataset_" + MODELNAME + "@" + FREQ + "x" + strstart_N + "~20000.csv"

    ctx = mp.get_context("spawn")
    skipped_records = []

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["i", "workload_hash", "trace_hash", "n_stores", "lat_mean_ms", "avg_power_w"]
        header += [f"f{k}" for k in range(656)]
        w.writerow(header)
        f.flush()

        for abs_idx in recs:
            result_queue = ctx.Queue()
            p = ctx.Process(target=process_one_record, args=(abs_idx, db_work_dir, result_queue))
            p.start()
            p.join(TIMEOUT_SECONDS)

            if p.is_alive():
                p.terminate()
                p.join()
                print(f"[{abs_idx} / {N}] TIMEOUT after {TIMEOUT_SECONDS}s, skipping record.")
                skipped_records.append((abs_idx, "timeout"))
                result_queue.close()
                result_queue.join_thread()
                continue

            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                print(f"[{abs_idx} / {N}] No result returned, skipping record.")
                skipped_records.append((abs_idx, "no_result"))
                result_queue.close()
                result_queue.join_thread()
                continue

            result_queue.close()
            result_queue.join_thread()

            if result[0] == "ok":
                _, row, workload_hash, lat_ms, avg_power = result
                w.writerow(row)
                f.flush()
                print(f"[{abs_idx} / {N}] Hash: {workload_hash} | Lat: {lat_ms:.3f}ms | Pwr: {avg_power:.1f}W")
            else:
                _, failed_idx, err = result
                print(f"[{failed_idx} / {N}] Schedule failed to build/run: {err}")
                skipped_records.append((failed_idx, f"error: {err}"))

    print("wrote:", out_path)

    print("\n[INFO] Skipped records summary:")
    print(f"[INFO] Total skipped: {len(skipped_records)}")
    if skipped_records:
        for idx, reason in skipped_records:
            print(f"  record {idx}: {reason}")
    else:
        print("  None")


if __name__ == "__main__":
    main()
import csv
import numpy as np
import tvm
from tvm import tir
from tvm import meta_schedule as ms
import hashlib
import subprocess
import atexit
import multiprocessing as mp
import queue


# # === CLOCK LOCKING FOR WSL -> WINDOWS ===
# def lock_clocks_windows(freq=2618):
#     print(f"\n[INFO] Attempting to lock Windows GPU clocks to {freq} MHz...")
#     cmd = f"Start-Process nvidia-smi.exe -ArgumentList '-i 0 -lgc {freq},{freq}' -Verb RunAs -Wait"
#     try:
#         subprocess.run(["powershell.exe", "-Command", cmd], check=True)
#         print("[INFO] GPU clocks locked successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"[WARN] Failed to lock clocks. Ensure you accepted the UAC prompt. Error: {e}")


# def unlock_clocks_windows():
#     print(f"\n[INFO] Attempting to unlock Windows GPU clocks...")
#     cmd = "Start-Process nvidia-smi.exe -ArgumentList '-i 0 -rgc' -Verb RunAs -Wait"
#     try:
#         subprocess.run(["powershell.exe", "-Command", cmd], check=True)
#         print("[INFO] GPU clocks unlocked successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"[WARN] Failed to unlock clocks. Error: {e}")


# # Ensure clocks unlock even if script crashes
# atexit.register(unlock_clocks_windows)
# # ========================================


def as_float_metric(x):
    if hasattr(x, "ratio"):
        return float(x.ratio)
    return float(x)


def make_nd(shape, dtype, dev):
    x = np.random.rand(*[int(s) for s in shape]).astype(dtype)
    return tvm.nd.array(x, device=dev)


def trace_fingerprint(trace):
    obj = trace.as_python() if hasattr(trace, "as_python") else repr(trace)
    if isinstance(obj, str):
        s = obj
    else:
        s = "\n".join(str(x) for x in obj)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# HYPERPARAMETERS
N = 20000
start_N = 4000
MODELNAME = "deepseekr1-qwen-14b"
FREQ = "2295"
TIMEOUT_SECONDS = 60
# FREQ = 2618 # Base clock lock freq


def process_one_record(abs_idx, db_work_dir, result_queue):
    try:
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

        agg = np.concatenate([feat.mean(0), feat.std(0), feat.min(0), feat.max(0)])

        rt_mod = tvm.build(sch.mod, target=target)
        args = [make_nd(t.shape, t.dtype, dev) for t in r.args_info]

        ftimer = rt_mod.time_evaluator("main", dev, number=1, repeat=1, min_repeat_ms=1000)
        timing = ftimer(*args)

        nvml = tvm.get_global_func("runtime.profiling.get_last_nvml_metrics")()
        avg_power = as_float_metric(nvml["avg_power_w"])

        row = [
            abs_idx,
            workload_hash,
            trace_hash,
            int(feat.shape[0]),
            float(timing.mean) * 1e3,
            float(avg_power),
        ]
        row += [float(x) for x in agg.tolist()]

        result_queue.put(
            (
                "ok",
                row,
                workload_hash,
                float(timing.mean) * 1e3,
                float(avg_power),
            )
        )

    except Exception as e:
        result_queue.put(("err", abs_idx, str(e)))


def main():
    # # Lock clocks before starting measurements
    # lock_clocks_windows(freq=FREQ)

    db_work_dir = "complete_tuning_logs_20000/tuning_logs_" + MODELNAME
    db = ms.database.JSONDatabase(work_dir=db_work_dir)
    all_recs = db.get_all_tuning_records()

    actual_N = min(N, len(all_recs))
    recs = [i for i in range(start_N, actual_N)]
    print(f"[INFO] Loaded {len(recs)} tuning records from database.")

    strstart_N = str(start_N)
    out_path = "dataset_" + MODELNAME + "@" + FREQ + "x" + strstart_N + "~20000.csv"

    ctx = mp.get_context("spawn")
    skipped_records = []

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["i", "workload_hash", "trace_hash", "n_stores", "lat_mean_ms", "avg_power_w"]
        header += [f"f{k}" for k in range(656)]
        w.writerow(header)
        f.flush()

        for abs_idx in recs:
            result_queue = ctx.Queue()
            p = ctx.Process(target=process_one_record, args=(abs_idx, db_work_dir, result_queue))
            p.start()
            p.join(TIMEOUT_SECONDS)

            if p.is_alive():
                p.terminate()
                p.join()
                print(f"[{abs_idx} / {N}] TIMEOUT after {TIMEOUT_SECONDS}s, skipping record.")
                skipped_records.append((abs_idx, "timeout"))
                result_queue.close()
                result_queue.join_thread()
                continue

            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                print(f"[{abs_idx} / {N}] No result returned, skipping record.")
                skipped_records.append((abs_idx, "no_result"))
                result_queue.close()
                result_queue.join_thread()
                continue

            result_queue.close()
            result_queue.join_thread()

            if result[0] == "ok":
                _, row, workload_hash, lat_ms, avg_power = result
                w.writerow(row)
                f.flush()
                print(f"[{abs_idx} / {N}] Hash: {workload_hash} | Lat: {lat_ms:.3f}ms | Pwr: {avg_power:.1f}W")
            else:
                _, failed_idx, err = result
                print(f"[{failed_idx} / {N}] Schedule failed to build/run: {err}")
                skipped_records.append((failed_idx, f"error: {err}"))

    print("wrote:", out_path)

    print("\n[INFO] Skipped records summary:")
    print(f"[INFO] Total skipped: {len(skipped_records)}")
    if skipped_records:
        for idx, reason in skipped_records:
            print(f"  record {idx}: {reason}")
    else:
        print("  None")


if __name__ == "__main__":
    main()
