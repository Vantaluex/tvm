import csv
import numpy as np
import tvm
from tvm import tir
from tvm import meta_schedule as ms
import hashlib

dev = tvm.cuda(0)

def as_float_metric(x):
    if hasattr(x, "ratio"):
        return float(x.ratio)
    return float(x)

def make_nd(shape, dtype):
    x = np.random.rand(*[int(s) for s in shape]).astype(dtype)
    return tvm.nd.array(x, device=dev)

def trace_fingerprint(trace):
    obj = trace.as_python() if hasattr(trace, "as_python") else repr(trace)

    # In some TVM builds, as_python() returns tvm.ir.container.Array (iterable of lines). [web:21]
    if isinstance(obj, str):
        s = obj
    else:
        # Works for tvm.ir.container.Array, Python list, tuple, etc.
        s = "\n".join(str(x) for x in obj)  # join lines into one string [web:156]

    return hashlib.sha1(s.encode("utf-8")).hexdigest()

N = 20 # HYPERPARAMETERS dataset amount
db = ms.database.JSONDatabase(work_dir="tuning_logs")
recs = db.get_all_tuning_records()[:N]
extractor = ms.feature_extractor.PerStoreFeature()

out_path = "eyas_gpu_dataset_200.csv"
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    header = ["i", "workload_hash", "trace_hash", "n_stores", "lat_mean_ms", "avg_power_w"]
    header += [f"f{k}" for k in range(656)]
    w.writerow(header)

    for i, r in enumerate(recs):
        mod = r.workload.mod
        target = r.target

        workload_hash = int(tvm.ir.structural_hash(mod))  # IRModule supports this [web:113]
        trace_hash = trace_fingerprint(r.trace)

        sch = tir.Schedule(mod, debug_mask="all")
        r.trace.apply_to_schedule(sch, remove_postproc=True)

        cand = ms.MeasureCandidate(sch=sch, args_info=r.args_info)
        ctx = ms.TuneContext(mod=mod, target=target, task_name=f"rec_{i}")
        (feat_nd,) = extractor.extract_from(ctx, candidates=[cand])
        feat = feat_nd.numpy()
        agg = np.concatenate([feat.mean(0), feat.std(0), feat.min(0), feat.max(0)])  # Eyas agg [file:3]

        rt_mod = tvm.build(sch.mod, target=target)
        args = [make_nd(t.shape, t.dtype) for t in r.args_info]

        ftimer = rt_mod.time_evaluator("main", dev, number=1, repeat=3, min_repeat_ms=3000) # HYPERPARAMETERS
        timing = ftimer(*args)

        nvml = tvm.get_global_func("runtime.profiling.get_last_nvml_metrics")()
        avg_power = as_float_metric(nvml["avg_power_w"])

        row = [i, workload_hash, trace_hash, int(feat.shape[0]), float(timing.mean) * 1e3, float(avg_power)]
        row += [float(x) for x in agg.tolist()]
        w.writerow(row)

print("wrote:", out_path)
