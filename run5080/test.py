import os
import numpy as np
import torch
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_fx
from tvm import dlight as dl

# ──────────────────────────────────────────────
# Optional: wrapper for HuggingFace models
# ──────────────────────────────────────────────

def wrap_output_to_tuple(m):
    class Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        # For BERT-like models: two named args
        def forward(self, input_ids, attention_mask):
            out = self.inner(input_ids, attention_mask)
            if hasattr(out, "last_hidden_state"):
                return (out.last_hidden_state,)
            if hasattr(out, "logits"):
                return (out.logits,)
            return tuple(out.values())
    return Wrapped(m).eval()

# ──────────────────────────────────────────────
# Single, unified model registry
# ──────────────────────────────────────────────

def get_model_and_input_specs(name, batch=32, seq_len=128, dtype="float32"):
    if name == "resnet50":
        from torchvision.models.resnet import ResNet50_Weights, resnet50
        m = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
        example_args = (torch.randn(batch, 3, 224, 224),)
        input_specs = [((batch, 3, 224, 224), dtype)]
        return m, example_args, input_specs

    if name == "mobilenet_v3_large":
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
        m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).eval()
        example_args = (torch.randn(batch, 3, 224, 224),)
        input_specs = [((batch, 3, 224, 224), dtype)]
        return m, example_args, input_specs

    if name == "densenet169":
        from torchvision.models import DenseNet169_Weights, densenet169
        m = densenet169(weights=DenseNet169_Weights.DEFAULT).eval()
        example_args = (torch.randn(batch, 3, 224, 224),)
        input_specs = [((batch, 3, 224, 224), dtype)]
        return m, example_args, input_specs

    if name == "bert":
        from transformers import BertModel, BertConfig
        config = BertConfig()
        m = BertModel(config).eval()
        m = wrap_output_to_tuple(m)

        token_ids = torch.randint(0, config.vocab_size, (batch, seq_len), dtype=torch.int64)
        attn_mask = torch.ones((batch, seq_len), dtype=torch.int64)

        example_args = (token_ids, attn_mask)
        input_specs = [((batch, seq_len), "int64"), ((batch, seq_len), "int64")]
        return m, example_args, input_specs

    raise ValueError(f"Unknown model name: {name}")

# ──────────────────────────────────────────────
# Compile + run
# ──────────────────────────────────────────────

def compile_and_run(model_name, target, dev, total_trials=20000, batch=32):
    m, example_args, input_specs = get_model_and_input_specs(model_name, batch=batch)

    with torch.no_grad():
        traced = torch.fx.symbolic_trace(m)

    mod = from_fx(traced, input_specs, keep_params_as_input=True)
    mod, params = relax.frontend.detach_params(mod)

    print(f"[{model_name}] num params: {len(params['main'])}")

    IS_IN_CI = os.getenv("CI", "") == "true"
    if not IS_IN_CI:
        mod = relax.get_pipeline(
            "static_shape_tuning",
            target=target,
            total_trials=total_trials,
            max_trials_per_task=200, 
            work_dir="tuning_logs", 
        )(mod)

        with target:
            # Avoid DLight matmul rule for now; fallback is safer for fused conv nets
            mod = dl.ApplyDefaultSchedule(
                # dl.gpu.Matmul(),
                dl.gpu.Fallback(),
            )(mod)

    ex = relax.build(mod, target=target)
    vm = relax.VirtualMachine(ex, dev)

    tvm_inputs = []
    for a in example_args:
        np_a = a.cpu().numpy()
        tvm_inputs.append(tvm.nd.array(np_a, dev))

    def to_dev(p):
        if isinstance(p, tvm.runtime.NDArray):
            return tvm.nd.array(p.numpy(), dev)
        return tvm.nd.array(p, dev)

    gpu_params = [to_dev(p) for p in params["main"]]

    vm.save_function("main", "main_saved", *tvm_inputs, *gpu_params)
    ft = vm.time_evaluator("main_saved", dev, number=50, repeat=3, min_repeat_ms=500)
    timing = ft()
    print(f"[{model_name}] timing: {timing}")

    out = vm["main"](*tvm_inputs, *gpu_params)
    if isinstance(out, tvm.runtime.NDArray):
        print(f"[{model_name}] out shape: {out.shape}, dtype: {out.dtype}")
    else:
        print(f"[{model_name}] out[0] shape: {out[0].shape}, dtype: {out[0].dtype}")

    return timing, out

# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

dev = tvm.cuda(0)
target = tvm.target.Target.from_device(dev)

timing, out = compile_and_run("mobilenet_v3_large", target, dev, total_trials=20000, batch=32)
print("Finished!")
