import os
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_fx
import tvm.meta_schedule as ms

from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    AutoModelForSemanticSegmentation,
    AutoModelForUniversalSegmentation,
)

# ============================================================================
# CONFIGURATION: EDIT THESE FIRST
# ============================================================================

IS_IN_CI = os.getenv("CI", "") == "true"

TOTAL_TRIALS = 64
BATCH_SIZE = 4
SEQ_LEN = 512
IMAGE_SIZE = 224

SELECTED_MODELS = [
    "rtdetr_r50",
    "segformer_b2",
    "convnextv2_tiny",
    "mask2former_swin_small",
    "deepseekr1_qwen_14b",
    "qwen2_5_3b",
    "llama_3_1_8b",
    "deberta_v3_base",
    "modernbert_base",
]

# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODEL_SPECS = {
    "deepseekr1_qwen_14b": {
        "kind": "manual_decoder",
        "model_name": "deepseekr1-qwen-14b",
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "batch_size": 4,
        "seq_len": 512,
    },
    "qwen2_5_3b": {
        "kind": "manual_decoder",
        "model_name": "qwen2.5-3b",
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "max_position_embeddings": 32768,
        "batch_size": 4,
        "seq_len": 512,
    },
    "llama_3_1_8b": {
        "kind": "manual_decoder",
        "model_name": "llama-3.1-8b",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "batch_size": 4,
        "seq_len": 512,
    },
    "deberta_v3_base": {
        "kind": "manual_encoder_from_hf_config",
        "hf_id": "microsoft/deberta-v3-base",
        "model_name": "deberta-v3-base",
        "batch_size": 8,
        "seq_len": 256,
    },
    "modernbert_base": {
        "kind": "manual_encoder_from_hf_config",
        "hf_id": "answerdotai/ModernBERT-base",
        "model_name": "modernbert-base",
        "batch_size": 8,
        "seq_len": 512,
    },
    "convnextv2_tiny": {
        "kind": "captured_image_classification",
        "hf_id": "facebook/convnextv2-tiny-22k-224",
        "model_name": "convnextv2-tiny",
        "batch_size": 8,
        "image_size": 224,
    },
    "rtdetr_r50": {
        "kind": "captured_object_detection",
        "hf_id": "PekingU/rtdetr_r50vd",
        "model_name": "rtdetr-r50",
        "batch_size": 2,
        "image_size": 640,
    },
    "mask2former_swin_small": {
        "kind": "captured_universal_segmentation",
        "hf_id": "facebook/mask2former-swin-small-cityscapes-semantic",
        "model_name": "mask2former-swin-small",
        "batch_size": 2,
        "image_size": 512,
    },
    "segformer_b2": {
        "kind": "captured_semantic_segmentation",
        "hf_id": "nvidia/segformer-b2-finetuned-ade-512-512",
        "model_name": "segformer-b2",
        "batch_size": 2,
        "image_size": 512,
    },
}

# ============================================================================
# CONFIG HELPERS
# ============================================================================

def make_manual_decoder_cfg(spec):
    return {
        "model_name": spec["model_name"],
        "hidden_size": spec["hidden_size"],
        "intermediate_size": spec["intermediate_size"],
        "num_attention_heads": spec["num_attention_heads"],
        "num_key_value_heads": spec["num_key_value_heads"],
        "max_position_embeddings": spec["max_position_embeddings"],
        "batch_size": spec.get("batch_size", BATCH_SIZE),
        "seq_len": spec.get("seq_len", SEQ_LEN),
    }


def make_encoder_cfg_from_hf(spec):
    hf_cfg = AutoConfig.from_pretrained(spec["hf_id"])
    return {
        "model_name": spec["model_name"],
        "hidden_size": int(hf_cfg.hidden_size),
        "intermediate_size": int(hf_cfg.intermediate_size),
        "num_attention_heads": int(hf_cfg.num_attention_heads),
        "num_key_value_heads": int(
            getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads)
        ),
        "max_position_embeddings": int(
            getattr(hf_cfg, "max_position_embeddings", 4096)
        ),
        "batch_size": spec.get("batch_size", BATCH_SIZE),
        "seq_len": spec.get("seq_len", SEQ_LEN),
    }


def print_cfg(cfg):
    print(f'Using config for {cfg["model_name"]}')
    print(f'HIDDEN_SIZE={cfg["hidden_size"]}')
    print(f'INTERMEDIATE_SIZE={cfg["intermediate_size"]}')
    print(f'NUM_ATTENTION_HEADS={cfg["num_attention_heads"]}')
    print(f'NUM_KEY_VALUE_HEADS={cfg["num_key_value_heads"]}')
    print(f'MAX_POSITION_EMBEDDINGS={cfg["max_position_embeddings"]}')
    print(f'HEAD_DIM={cfg["hidden_size"] // cfg["num_attention_heads"]}')
    print(f'BATCH_SIZE={cfg["batch_size"]}')
    print(f'SEQ_LEN={cfg["seq_len"]}')


# ============================================================================
# TASK MERGING
# ============================================================================

def task_structural_key(task):
    mod_hash = tvm.ir.structural_hash(task.mod)
    target_key = str(task.target)
    return (task.task_name, mod_hash, target_key)


def merge_extracted_tasks(extracted_tasks):
    buckets = defaultdict(list)

    for task in extracted_tasks:
        key = task_structural_key(task)
        merged = False

        for existing in buckets[key]:
            same_mod = tvm.ir.structural_equal(task.mod, existing.mod)
            same_target = str(task.target) == str(existing.target)
            same_name = task.task_name == existing.task_name

            if same_mod and same_target and same_name:
                existing.weight += task.weight
                merged = True
                break

        if not merged:
            buckets[key].append(
                ms.ExtractedTask(
                    task_name=task.task_name,
                    mod=task.mod,
                    target=task.target,
                    dispatched=task.dispatched,
                    weight=task.weight,
                )
            )

    merged_tasks = []
    for group in buckets.values():
        merged_tasks.extend(group)
    return merged_tasks


def print_task_merge_stats(before_tasks, after_tasks):
    before_weight = sum(int(task.weight) for task in before_tasks)
    after_weight = sum(int(task.weight) for task in after_tasks)

    print("\n" + "-" * 80)
    print("Task aggregation summary")
    print(f"Unique tasks before merge: {len(before_tasks)}")
    print(f"Unique tasks after merge:  {len(after_tasks)}")
    print(f"Total task weight before:  {before_weight}")
    print(f"Total task weight after:   {after_weight}")
    print("-" * 80)

    if len(after_tasks) > 0:
        sorted_tasks = sorted(
            after_tasks,
            key=lambda t: int(t.weight),
            reverse=True,
        )
        print("Top merged tasks by weight:")
        for task in sorted_tasks[:20]:
            print(
                f"  name={task.task_name:<20} "
                f"weight={int(task.weight):<4} "
                f"hash={tvm.ir.structural_hash(task.mod)}"
            )
        print("-" * 80)


# ============================================================================
# DECODER-STYLE OPERATOR DEFINITIONS
# ============================================================================

class AttentionQKVProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        total_qkv_dim = (
            cfg["num_attention_heads"] + 2 * cfg["num_key_value_heads"]
        ) * head_dim
        self.qkv_proj = nn.Linear(cfg["hidden_size"], total_qkv_dim, bias=True)

    def forward(self, hidden_states):
        return self.qkv_proj(hidden_states)


class AttentionMatmul(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]

    def forward(self, query, key):
        batch, num_kv_heads, seq_len, head_dim = key.shape
        num_query_heads = query.shape[1]
        repeats = num_query_heads // num_kv_heads
        key = key.unsqueeze(2).expand(-1, -1, repeats, -1, -1).reshape(
            batch, num_query_heads, seq_len, head_dim
        )
        key_transposed = key.transpose(-2, -1)
        return torch.matmul(query, key_transposed) / (self.head_dim ** 0.5)


class AttentionSoftmax(nn.Module):
    def forward(self, attention_scores):
        return F.softmax(attention_scores, dim=-1)


class AttentionValueMatmul(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, attention_weights, value):
        batch, num_kv_heads, seq_len, head_dim = value.shape
        num_query_heads = attention_weights.shape[1]
        repeats = num_query_heads // num_kv_heads
        value = value.unsqueeze(2).expand(-1, -1, repeats, -1, -1).reshape(
            batch, num_query_heads, seq_len, head_dim
        )
        return torch.matmul(attention_weights, value)


class AttentionOutputProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.o_proj = nn.Linear(cfg["hidden_size"], cfg["hidden_size"], bias=False)

    def forward(self, hidden_states):
        return self.o_proj(hidden_states)


class MLPGateUpProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            cfg["hidden_size"], 2 * cfg["intermediate_size"], bias=False
        )

    def forward(self, hidden_states):
        return self.gate_up_proj(hidden_states)


class MLPActivation(nn.Module):
    def forward(self, gate_up_states):
        gate, up = gate_up_states.chunk(2, dim=-1)
        return F.silu(gate) * up


class MLPDownProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.down_proj = nn.Linear(
            cfg["intermediate_size"], cfg["hidden_size"], bias=False
        )

    def forward(self, hidden_states):
        return self.down_proj(hidden_states)


class RMSNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg["hidden_size"]))
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


# ============================================================================
# ENCODER-STYLE OPERATOR DEFINITIONS
# ============================================================================

class QProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.q_proj = nn.Linear(cfg["hidden_size"], cfg["hidden_size"], bias=True)

    def forward(self, hidden_states):
        return self.q_proj(hidden_states)


class KProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.k_proj = nn.Linear(cfg["hidden_size"], cfg["hidden_size"], bias=True)

    def forward(self, hidden_states):
        return self.k_proj(hidden_states)


class VProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.v_proj = nn.Linear(cfg["hidden_size"], cfg["hidden_size"], bias=True)

    def forward(self, hidden_states):
        return self.v_proj(hidden_states)


class EncoderAttentionMatmul(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]

    def forward(self, query, key):
        key_transposed = key.transpose(-2, -1)
        return torch.matmul(query, key_transposed) / (self.head_dim ** 0.5)


class EncoderAttentionValueMatmul(nn.Module):
    def forward(self, attention_weights, value):
        return torch.matmul(attention_weights, value)


class MLPDenseIn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense_in = nn.Linear(
            cfg["hidden_size"], cfg["intermediate_size"], bias=True
        )

    def forward(self, hidden_states):
        return self.dense_in(hidden_states)


class MLPActivationGELU(nn.Module):
    def forward(self, hidden_states):
        return F.gelu(hidden_states)


class LayerNormOp(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm = nn.LayerNorm(cfg["hidden_size"])

    def forward(self, hidden_states):
        return self.layer_norm(hidden_states)


# ============================================================================
# EXTRACTION HELPERS
# ============================================================================

def extract_tasks_for_operator(model, input_shapes, operator_name, target):
    print(f"Tracing and Extracting operator: {operator_name}")

    model = copy.deepcopy(model).cpu().eval()

    with torch.no_grad():
        traced = torch.fx.symbolic_trace(model)

    input_specs = [(shape, "float32") for shape in input_shapes]
    mod = from_fx(traced, input_specs, keep_params_as_input=True)
    mod = relax.transform.LegalizeOps()(mod)
    mod, _ = relax.frontend.detach_params(mod)

    tasks = ms.relax_integration.extract_tasks(mod, target)
    return tasks


# ============================================================================
# MANUAL OPERATOR BANKS
# ============================================================================

def build_decoder_operators(cfg):
    batch_size = cfg["batch_size"]
    seq_len = cfg["seq_len"]
    hidden_size = cfg["hidden_size"]
    intermediate_size = cfg["intermediate_size"]
    num_attention_heads = cfg["num_attention_heads"]
    num_key_value_heads = cfg["num_key_value_heads"]
    head_dim = hidden_size // num_attention_heads

    return [
        ("attention_qkv_projection", AttentionQKVProjection(cfg), [(batch_size, seq_len, hidden_size)]),
        ("attention_matmul", AttentionMatmul(cfg), [
            (batch_size, num_attention_heads, seq_len, head_dim),
            (batch_size, num_key_value_heads, seq_len, head_dim),
        ]),
        ("attention_softmax", AttentionSoftmax(), [
            (batch_size, num_attention_heads, seq_len, seq_len)
        ]),
        ("attention_value_matmul", AttentionValueMatmul(cfg), [
            (batch_size, num_attention_heads, seq_len, seq_len),
            (batch_size, num_key_value_heads, seq_len, head_dim),
        ]),
        ("attention_output_projection", AttentionOutputProjection(cfg), [(batch_size, seq_len, hidden_size)]),
        ("mlp_gate_up_projection", MLPGateUpProjection(cfg), [(batch_size, seq_len, hidden_size)]),
        ("mlp_activation", MLPActivation(), [(batch_size, seq_len, 2 * intermediate_size)]),
        ("mlp_down_projection", MLPDownProjection(cfg), [(batch_size, seq_len, intermediate_size)]),
        ("rms_norm", RMSNorm(cfg), [(batch_size, seq_len, hidden_size)]),
    ]


def build_encoder_operators(cfg):
    batch_size = cfg["batch_size"]
    seq_len = cfg["seq_len"]
    hidden_size = cfg["hidden_size"]
    intermediate_size = cfg["intermediate_size"]
    num_attention_heads = cfg["num_attention_heads"]
    head_dim = hidden_size // num_attention_heads

    return [
        ("q_projection", QProjection(cfg), [(batch_size, seq_len, hidden_size)]),
        ("k_projection", KProjection(cfg), [(batch_size, seq_len, hidden_size)]),
        ("v_projection", VProjection(cfg), [(batch_size, seq_len, hidden_size)]),
        ("attention_matmul", EncoderAttentionMatmul(cfg), [
            (batch_size, num_attention_heads, seq_len, head_dim),
            (batch_size, num_attention_heads, seq_len, head_dim),
        ]),
        ("attention_softmax", AttentionSoftmax(), [
            (batch_size, num_attention_heads, seq_len, seq_len)
        ]),
        ("attention_value_matmul", EncoderAttentionValueMatmul(), [
            (batch_size, num_attention_heads, seq_len, seq_len),
            (batch_size, num_attention_heads, seq_len, head_dim),
        ]),
        ("attention_output_projection", AttentionOutputProjection(cfg), [(batch_size, seq_len, hidden_size)]),
        ("mlp_dense_in", MLPDenseIn(cfg), [(batch_size, seq_len, hidden_size)]),
        ("mlp_activation_gelu", MLPActivationGELU(), [(batch_size, seq_len, intermediate_size)]),
        ("mlp_down_projection", MLPDownProjection(cfg), [(batch_size, seq_len, intermediate_size)]),
        ("layer_norm", LayerNormOp(cfg), [(batch_size, seq_len, hidden_size)]),
    ]


# ============================================================================
# CAPTURE-BASED VISION MODEL SUPPORT
# ============================================================================

def is_leaf_module(module):
    return len(list(module.children())) == 0


def should_capture_module(module):
    capture_types = (
        nn.Conv2d,
        nn.Linear,
        nn.LayerNorm,
        nn.GELU,
        nn.SiLU,
        nn.ReLU,
        nn.MaxPool2d,
        nn.AvgPool2d,
        nn.AdaptiveAvgPool2d,
    )
    return is_leaf_module(module) and isinstance(module, capture_types)


def get_first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for item in x:
            out = get_first_tensor(item)
            if out is not None:
                return out
    if isinstance(x, dict):
        for _, item in x.items():
            out = get_first_tensor(item)
            if out is not None:
                return out
    return None


def normalize_tensor_args(args):
    tensor_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensor_args.append(arg.detach().cpu())
    return tensor_args


def build_pixel_inputs(batch_size, image_size):
    return {
        "pixel_values": torch.randn(
            batch_size, 3, image_size, image_size, dtype=torch.float32
        )
    }


def load_captured_model(spec):
    kind = spec["kind"]
    hf_id = spec["hf_id"]
    batch_size = spec.get("batch_size", BATCH_SIZE)
    image_size = spec.get("image_size", IMAGE_SIZE)

    if kind == "captured_image_classification":
        model = AutoModelForImageClassification.from_pretrained(hf_id).cpu().eval()
        inputs = build_pixel_inputs(batch_size, image_size)
        return model, inputs

    if kind == "captured_object_detection":
        model = AutoModelForObjectDetection.from_pretrained(hf_id).cpu().eval()
        inputs = build_pixel_inputs(batch_size, image_size)
        return model, inputs

    if kind == "captured_semantic_segmentation":
        model = AutoModelForSemanticSegmentation.from_pretrained(hf_id).cpu().eval()
        inputs = build_pixel_inputs(batch_size, image_size)
        return model, inputs

    if kind == "captured_universal_segmentation":
        model = AutoModelForUniversalSegmentation.from_pretrained(hf_id).cpu().eval()
        inputs = build_pixel_inputs(batch_size, image_size)
        return model, inputs

    raise ValueError(f"Unsupported captured model kind: {kind}")


def capture_leaf_operators(model, example_inputs):
    captured = []
    seen = set()
    hooks = []

    def make_hook(module_name, module_ref):
        def hook(_module, args, output):
            if module_name in seen:
                return

            tensor_args = normalize_tensor_args(args)
            out = get_first_tensor(output)

            if len(tensor_args) == 0 or out is None:
                return

            input_shapes = [tuple(int(v) for v in t.shape) for t in tensor_args]
            captured.append(
                (module_name, copy.deepcopy(module_ref).cpu().eval(), input_shapes)
            )
            seen.add(module_name)

        return hook

    for name, module in model.named_modules():
        if should_capture_module(module):
            hooks.append(module.register_forward_hook(make_hook(name, module)))

    with torch.no_grad():
        _ = model(**example_inputs)

    for hook in hooks:
        hook.remove()

    captured.sort(key=lambda x: x[0])
    return captured


# ============================================================================
# OPERATOR BUILDER
# ============================================================================

def build_operators_for_model(model_key):
    spec = MODEL_SPECS[model_key]
    kind = spec["kind"]

    if kind == "manual_decoder":
        cfg = make_manual_decoder_cfg(spec)
        print_cfg(cfg)
        return build_decoder_operators(cfg), cfg["model_name"]

    if kind == "manual_encoder_from_hf_config":
        cfg = make_encoder_cfg_from_hf(spec)
        print_cfg(cfg)
        return build_encoder_operators(cfg), cfg["model_name"]

    if kind.startswith("captured_"):
        print(f"Loading captured model: {spec['model_name']} ({spec['hf_id']})")
        model, example_inputs = load_captured_model(spec)
        captured_operators = capture_leaf_operators(model, example_inputs)
        print(f"Captured {len(captured_operators)} leaf operators from {spec['model_name']}")
        return captured_operators, spec["model_name"]

    raise ValueError(f"Unsupported model kind: {kind}")


# ============================================================================
# ONE MODEL RUN
# ============================================================================

def run_one_model(model_key):
    print("\n" + "#" * 100)
    print(f"MODEL KEY: {model_key}")
    print("#" * 100)

    operators, model_name = build_operators_for_model(model_key)

    print("Starting Step 1: extraction and global tuning")
    print(f"Total Global Trial Budget: {TOTAL_TRIALS}")

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)

    all_global_tasks = []
    skipped_operators = []

    for i, (name, module, shapes) in enumerate(operators, 1):
        print(f"[{i}/{len(operators)}] {name}")
        try:
            extracted_tasks = extract_tasks_for_operator(module, shapes, name, target)
            all_global_tasks.extend(extracted_tasks)
        except Exception as e:
            skipped_operators.append((name, str(e)))
            print(f"[SKIP] {name} failed during tracing/extraction: {e}")

    print("\n" + "=" * 80)
    print(f"Extracted {len(all_global_tasks)} raw tasks before merge.")
    print(f"Skipped operators: {len(skipped_operators)}")
    print("=" * 80)

    merged_tasks = merge_extracted_tasks(all_global_tasks)
    print_task_merge_stats(all_global_tasks, merged_tasks)

    if len(merged_tasks) == 0:
        print("No tunable tasks remained after merging.")
        return

    if not IS_IN_CI:
        work_dir = "./tuning_logs_" + model_name

        tasks, task_weights = ms.relax_integration.extracted_tasks_to_tune_contexts(
            extracted_tasks=merged_tasks,
            work_dir=work_dir,
        )

        print("\n" + "=" * 80)
        print(f"Found {len(tasks)} distinct tuning tasks after merge.")
        print("Passing control to TVM's Global TaskScheduler...")
        print("=" * 80)

        ms.tune_tasks(
            tasks=tasks,
            task_weights=task_weights,
            work_dir=work_dir,
            max_trials_global=TOTAL_TRIALS,
        )

        print(f"\nTuning completed. Log saved to {work_dir}")

    if skipped_operators:
        print("\nSkipped operator summary:")
        for name, err in skipped_operators[:50]:
            print(f"  {name}: {err}")
        if len(skipped_operators) > 50:
            print(f"  ... and {len(skipped_operators) - 50} more")


# ============================================================================
# MAIN
# ============================================================================

def main():
    for model_key in SELECTED_MODELS:
        run_one_model(model_key)


if __name__ == "__main__":
    main()
