import os
import numpy as np
import torch
from torchvision.models.resnet import ResNet50_Weights, resnet50

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_fx
from tvm import dlight as dl

# Torch model - NEW: ResNet50
m = resnet50(weights=ResNet50_Weights.DEFAULT).eval()

with torch.no_grad():
    traced = torch.fx.symbolic_trace(m)

# OLD (batch size too small)# Params as inputs (versatile) 
# mod = from_fx(traced, [((1, 3, 224, 224), "float32")], keep_params_as_input=True)
# mod, params = relax.frontend.detach_params(mod)
# # Use batch size 32 to create heavier TIR kernels

mod = from_fx(traced, [((32, 3, 224, 224), "float32")], keep_params_as_input=True)
mod, params = relax.frontend.detach_params(mod) # Put this back!

print("num params:", len(params["main"]))

dev = tvm.cuda(0)
target = tvm.target.Target.from_device(dev)

IS_IN_CI = os.getenv("CI", "") == "true"
TOTAL_TRIALS = 64

if not IS_IN_CI:
    mod = relax.get_pipeline(
        "static_shape_tuning",
        target=target,
        total_trials=TOTAL_TRIALS,
    )(mod)

    # Add default GPU schedules so TIR has proper thread bindings
    with target:
        mod = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(),
            dl.gpu.Fallback(),
        )(mod)

    ex = relax.build(mod, target=target)
    vm = relax.VirtualMachine(ex, dev)

    # FIX: Dummy data must match the batch size 32
    x = tvm.nd.array(np.random.rand(32, 3, 224, 224).astype("float32"), dev)

    def to_dev(p):
        if isinstance(p, tvm.runtime.NDArray):
            return tvm.nd.array(p.numpy(), dev)
        return tvm.nd.array(p, dev)

    gpu_params = [to_dev(p) for p in params["main"]]

    vm.save_function("main", "main_saved", x, *gpu_params)
    ft = vm.time_evaluator("main_saved", dev, number=50, repeat=3, min_repeat_ms=500)
    print("timing:", ft())

    print("nvml after time_evaluator:", tvm.get_global_func("runtime.profiling.get_last_nvml_metrics")())

    y = vm["main"](x, *gpu_params)

    if isinstance(y, tvm.runtime.NDArray):
        print("out:", y.shape, y.dtype, float(y.numpy().reshape(-1)[0]))
    else:
        y0 = y[0]
        print("out[0]:", y0.shape, y0.dtype, float(y0.numpy().reshape(-1)[0]))

print(tvm.get_global_func("runtime.profiling.get_last_nvml_metrics")())
