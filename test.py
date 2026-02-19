import os
import numpy as np
import tvm
from tvm import te

print("python pid:", os.getpid())

# Tiny CUDA kernel: B = A + 1
n = 1024
A = te.placeholder((n,), "float32", name="A")
B = te.compute((n,), lambda i: A[i] + tvm.tir.const(1.0, "float32"), name="B")
s = te.create_schedule(B.op)

f = tvm.build(s, [A, B], target="cuda")

dev = tvm.cuda(0)
a = tvm.nd.array(np.random.rand(n).astype("float32"), dev)
b = tvm.nd.array(np.zeros(n, dtype="float32"), dev)

ft = f.time_evaluator(f.entry_name, dev, number=200, repeat=5)
print("time:", ft(a, b))

get = tvm.get_global_func("runtime.profiling.get_last_nvml_metrics")
print("nvml:", get())
