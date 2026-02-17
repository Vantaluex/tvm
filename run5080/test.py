import numpy as np
import tvm
from tvm import tir
from tvm import meta_schedule as ms


def as_float_metric(x):
    # TVM profiling metrics like Ratio/Percent/Duration wrap a Python float internally. [web:100]
    if hasattr(x, "ratio"):
        return float(x.ratio)  # Ratio(ratio: float) [web:100]
    if hasattr(x, "percent"):
        return float(x.percent)
    if hasattr(x, "duration"):
        return float(x.duration)
    if hasattr(x, "count"):
        return float(x.count)
    return float(x)

dev = tvm.cuda(0)

db = ms.database.JSONDatabase(work_dir="tuning_logs")
recs = db.get_all_tuning_records()[:10]

extractor = ms.feature_extractor.PerStoreFeature()

def make_nd(shape, dtype):
    x = np.random.rand(*[int(s) for s in shape]).astype(dtype)
    return tvm.nd.array(x, device=dev)

for i, r in enumerate(recs):
    mod = r.workload.mod
    target = r.target

    sch = tir.Schedule(mod, debug_mask="all")
    r.trace.apply_to_schedule(sch, remove_postproc=True)

    # Eyas stage 1+2: per-store -> aggregate mean/std/min/max [file:3]
    cand = ms.MeasureCandidate(sch=sch, args_info=r.args_info)
    ctx = ms.TuneContext(mod=mod, target=target, task_name=f"rec_{i}")
    (feat_nd,) = extractor.extract_from(ctx, candidates=[cand])
    feat = feat_nd.numpy()
    agg = np.concatenate([feat.mean(0), feat.std(0), feat.min(0), feat.max(0)])

    # Build & run candidate
    rt_mod = tvm.build(sch.mod, target=target)
    args = [make_nd(t.shape, t.dtype) for t in r.args_info]

    ftimer = rt_mod.time_evaluator("main", dev, number=50, repeat=3, min_repeat_ms=500)  # [web:90]
    timing = ftimer(*args)

    nvml = tvm.get_global_func("runtime.profiling.get_last_nvml_metrics")()
    avg_power = as_float_metric(nvml["avg_power_w"])
    avg_clock = as_float_metric(nvml["avg_clock_mhz"])

    print(i, "stores:", feat.shape[0], "Nfeat:", feat.shape[1], "agg_len:", agg.shape[0],
          "lat_mean_ms:", float(timing.mean) * 1e3, "avg_power_w:", avg_power)
    