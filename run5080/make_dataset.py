import csv
import time
import queue
import hashlib
import multiprocessing as mp

import numpy as np
import tvm
from tvm import tir
from tvm import meta_schedule as ms


def as_float_metric(x):
    if hasattr(x, "ratio"):
        return float(x.ratio)
    return float(x)


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


# =========================
# HYPERPARAMETERS
# =========================
N = 20000
start_N = 0
MODELNAME = "deepseekr1-qwen-14b"
FREQ = "2295"

MIN_REPEAT_MS = 1500
NUMBER = 1
REPEAT = 1

PER_RECORD_TIMEOUT_SECONDS = 60
CHUNK_SIZE = 32
CHUNK_TIMEOUT_SECONDS = 600

DB_WORK_DIR = "complete_tuning_logs_20000/tuning_logs_" + MODELNAME
OUT_PATH = "dataset_" + MODELNAME + "@" + FREQ + "x" + str(start_N) + "~20000.csv"
SKIPPED_PATH = "skipped_" + MODELNAME + "@" + FREQ + "x" + str(start_N) + "~20000.txt"


def run_record(abs_idx, r):
    dev = tvm.cuda(0)

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

    ftimer = rt_mod.time_evaluator(
        "main",
        dev,
        number=NUMBER,
        repeat=REPEAT,
        min_repeat_ms=MIN_REPEAT_MS,
    )
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

    return (
        "ok",
        abs_idx,
        row,
        workload_hash,
        float(timing.mean) * 1e3,
        float(avg_power),
    )


def process_one_record(abs_idx, db_work_dir, result_queue):
    try:
        db = ms.database.JSONDatabase(work_dir=db_work_dir)
        all_recs = db.get_all_tuning_records()
        r = all_recs[abs_idx]
        result_queue.put(run_record(abs_idx, r))
    except Exception as e:
        result_queue.put(("err", abs_idx, str(e)))


def process_chunk(chunk_indices, db_work_dir, result_queue):
    try:
        db = ms.database.JSONDatabase(work_dir=db_work_dir)
        all_recs = db.get_all_tuning_records()

        for abs_idx in chunk_indices:
            try:
                r = all_recs[abs_idx]
                result_queue.put(run_record(abs_idx, r))
            except Exception as e:
                result_queue.put(("err", abs_idx, str(e)))

        result_queue.put(("chunk_done",))
    except Exception as e:
        result_queue.put(("chunk_err", str(e)))


def chunkify(xs, chunk_size):
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]


def handle_result(result, writer, out_file, skipped_records, actual_N, finished_set):
    tag = result[0]

    if tag == "ok":
        _, abs_idx, row, workload_hash, lat_ms, avg_power = result
        writer.writerow(row)
        out_file.flush()
        finished_set.add(abs_idx)
        print(f"[{abs_idx} / {actual_N}] Hash: {workload_hash} | Lat: {lat_ms:.3f}ms | Pwr: {avg_power:.1f}W")
        return

    if tag == "err":
        _, failed_idx, err = result
        finished_set.add(failed_idx)
        skipped_records.append((failed_idx, f"error: {err}"))
        print(f"[{failed_idx} / {actual_N}] Schedule failed to build/run: {err}")
        return


def drain_queue_nonblocking(result_queue, writer, out_file, skipped_records, actual_N, finished_set):
    while True:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            break

        if result[0] == "chunk_done":
            continue
        if result[0] == "chunk_err":
            print(f"[WARN] Chunk worker failed: {result[1]}")
            continue

        handle_result(result, writer, out_file, skipped_records, actual_N, finished_set)


def run_one_record_with_timeout(ctx, abs_idx, writer, out_file, skipped_records, actual_N):
    result_queue = ctx.Queue()
    p = ctx.Process(target=process_one_record, args=(abs_idx, DB_WORK_DIR, result_queue))
    p.start()
    p.join(PER_RECORD_TIMEOUT_SECONDS)

    if p.is_alive():
        p.terminate()
        p.join()
        print(f"[{abs_idx} / {actual_N}] TIMEOUT after {PER_RECORD_TIMEOUT_SECONDS}s, skipping record.")
        skipped_records.append((abs_idx, "timeout"))
        result_queue.close()
        result_queue.join_thread()
        return

    try:
        result = result_queue.get(timeout=5)
    except queue.Empty:
        print(f"[{abs_idx} / {actual_N}] No result returned, skipping record.")
        skipped_records.append((abs_idx, "no_result"))
        result_queue.close()
        result_queue.join_thread()
        return

    result_queue.close()
    result_queue.join_thread()

    finished_set = set()
    handle_result(result, writer, out_file, skipped_records, actual_N, finished_set)


def run_chunk_with_resume(ctx, chunk, writer, out_file, skipped_records, actual_N):
    result_queue = ctx.Queue()
    p = ctx.Process(target=process_chunk, args=(chunk, DB_WORK_DIR, result_queue))
    p.start()

    start_t = time.time()
    finished_in_chunk = set()
    chunk_done = False

    while True:
        remaining = CHUNK_TIMEOUT_SECONDS - (time.time() - start_t)
        if remaining <= 0:
            break

        try:
            result = result_queue.get(timeout=min(1.0, max(0.1, remaining)))
        except queue.Empty:
            if not p.is_alive():
                break
            continue

        if result[0] == "chunk_done":
            chunk_done = True
            break

        if result[0] == "chunk_err":
            print(f"[WARN] Chunk worker failed: {result[1]}")
            break

        handle_result(result, writer, out_file, skipped_records, actual_N, finished_in_chunk)

    drain_queue_nonblocking(result_queue, writer, out_file, skipped_records, actual_N, finished_in_chunk)

    if p.is_alive():
        p.terminate()
        p.join()
    else:
        p.join()

    drain_queue_nonblocking(result_queue, writer, out_file, skipped_records, actual_N, finished_in_chunk)

    result_queue.close()
    result_queue.join_thread()

    unfinished = [idx for idx in chunk if idx not in finished_in_chunk]

    if unfinished:
        print(f"[WARN] Chunk incomplete. Finished {len(finished_in_chunk)}/{len(chunk)}. Retrying unfinished individually: {unfinished}")
        for abs_idx in unfinished:
            run_one_record_with_timeout(ctx, abs_idx, writer, out_file, skipped_records, actual_N)
    else:
        print(f"[INFO] Chunk completed fully ({len(chunk)} records).")


def main():
    db = ms.database.JSONDatabase(work_dir=DB_WORK_DIR)
    all_recs = db.get_all_tuning_records()

    actual_N = min(N, len(all_recs))
    recs = list(range(start_N, actual_N))
    chunks = chunkify(recs, CHUNK_SIZE)

    print(f"[INFO] Loaded {len(recs)} tuning records from database.")
    print(f"[INFO] Chunk size: {CHUNK_SIZE}")
    print(f"[INFO] min_repeat_ms: {MIN_REPEAT_MS}")

    ctx = mp.get_context("spawn")
    skipped_records = []

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        header = ["i", "workload_hash", "trace_hash", "n_stores", "lat_mean_ms", "avg_power_w"]
        header += [f"f{k}" for k in range(656)]
        w.writerow(header)
        f.flush()

        for chunk_id, chunk in enumerate(chunks):
            print(f"[INFO] Starting chunk {chunk_id + 1}/{len(chunks)}: {chunk}")
            run_chunk_with_resume(ctx, chunk, w, f, skipped_records, actual_N)

    with open(SKIPPED_PATH, "w") as sf:
        for idx, reason in skipped_records:
            sf.write(f"{idx}\t{reason}\n")

    print("wrote:", OUT_PATH)
    print("wrote:", SKIPPED_PATH)
    print("\n[INFO] Skipped records summary:")
    print(f"[INFO] Total skipped: {len(skipped_records)}")
    if skipped_records:
        for idx, reason in skipped_records[:50]:
            print(f"  record {idx}: {reason}")
        if len(skipped_records) > 50:
            print(f"  ... and {len(skipped_records) - 50} more")
    else:
        print("  None")


if __name__ == "__main__":
    main()
