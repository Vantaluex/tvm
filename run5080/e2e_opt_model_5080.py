import os
import json
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_fx
import tvm.meta_schedule as ms

# ============================================================================
# CONFIGURATION
# ============================================================================
IS_IN_CI = os.getenv("CI", "") == "true"
TOTAL_TRIALS = 20000  # Set to your massive target budget
BATCH_SIZE = 4
SEQ_LEN = 512

# ============================================================================
# manually input model config from HuggingFace
# ============================================================================

MODEL_NAME = "deepseekr1-qwen-14b"

HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 13824
NUM_ATTENTION_HEADS = 40
NUM_KEY_VALUE_HEADS = 8
MAX_POSITION_EMBEDDINGS = 131072

HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS

print(f"Using manual config for {MODEL_NAME}")
print(f"HIDDEN_SIZE={HIDDEN_SIZE}")
print(f"INTERMEDIATE_SIZE={INTERMEDIATE_SIZE}")
print(f"NUM_ATTENTION_HEADS={NUM_ATTENTION_HEADS}")
print(f"NUM_KEY_VALUE_HEADS={NUM_KEY_VALUE_HEADS}")
print(f"MAX_POSITION_EMBEDDINGS={MAX_POSITION_EMBEDDINGS}")
print(f"HEAD_DIM={HEAD_DIM}")

# ============================================================================
# PyTorch Operator Definitions
# ============================================================================
class AttentionQKVProjection(nn.Module):
    def __init__(self):
        super().__init__()
        total_qkv_dim = (NUM_ATTENTION_HEADS + 2 * NUM_KEY_VALUE_HEADS) * HEAD_DIM
        self.qkv_proj = nn.Linear(HIDDEN_SIZE, total_qkv_dim, bias=True)
    def forward(self, hidden_states):
        return self.qkv_proj(hidden_states)

class AttentionMatmul(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key):
        batch, num_kv_heads, seq_len, head_dim = key.shape
        num_query_heads = query.shape[1]
        repeats = num_query_heads // num_kv_heads 
        key = key.unsqueeze(2).expand(-1, -1, repeats, -1, -1).reshape(batch, num_query_heads, seq_len, head_dim) 
        key_transposed = key.transpose(-2, -1)
        return torch.matmul(query, key_transposed) / (HEAD_DIM ** 0.5)
    
class AttentionSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, attention_scores):
        return F.softmax(attention_scores, dim=-1)

class AttentionValueMatmul(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, attention_weights, value):
        batch, num_kv_heads, seq_len, head_dim = value.shape
        num_query_heads = attention_weights.shape[1]
        repeats = num_query_heads // num_kv_heads
        value = value.unsqueeze(2).expand(-1, -1, repeats, -1, -1).reshape(batch, num_query_heads, seq_len, head_dim) 
        return torch.matmul(attention_weights, value)

class AttentionOutputProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
    def forward(self, hidden_states):
        return self.o_proj(hidden_states)

class MLPGateUpProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE, bias=False)
    def forward(self, hidden_states):
        return self.gate_up_proj(hidden_states)

class MLPActivation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, gate_up_states):
        gate, up = gate_up_states.chunk(2, dim=-1)
        return F.silu(gate) * up

class MLPDownProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)
    def forward(self, hidden_states):
        return self.down_proj(hidden_states)

class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(HIDDEN_SIZE))
        self.variance_epsilon = 1e-6
    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


# ============================================================================
# Step 1: Extract Tasks
# ============================================================================
def extract_tasks_for_operator(model_class, input_shapes, operator_name, target):
    """Traces the PyTorch model to TVM IR and pulls out the tunable tasks."""
    print(f"Tracing and Extracting: {operator_name}")
    
    model = model_class().eval()
    dummy_inputs = [torch.randn(*shape, dtype=torch.float32) for shape in input_shapes]
    with torch.no_grad():
        traced = torch.fx.symbolic_trace(model)
        
    input_specs = [(shape, "float32") for shape in input_shapes]
    mod = from_fx(traced, input_specs, keep_params_as_input=True)
    mod = relax.transform.LegalizeOps()(mod)
    mod, params = relax.frontend.detach_params(mod)
    
    # This pulls out the underlying TIR blocks (the actual math) into a MetaSchedule list
    tasks = ms.relax_integration.extract_tasks(mod, target)
    return tasks

# ============================================================================
# Main
# ============================================================================
def main():
    operators = [
        ("attention_qkv_projection", AttentionQKVProjection, [(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)]),
        ("attention_matmul", AttentionMatmul, [(BATCH_SIZE, NUM_ATTENTION_HEADS, SEQ_LEN, HEAD_DIM), (BATCH_SIZE, NUM_KEY_VALUE_HEADS, SEQ_LEN, HEAD_DIM)]),
        ("attention_softmax", AttentionSoftmax, [(BATCH_SIZE, NUM_ATTENTION_HEADS, SEQ_LEN, SEQ_LEN)]),
        ("attention_value_matmul", AttentionValueMatmul, [(BATCH_SIZE, NUM_ATTENTION_HEADS, SEQ_LEN, SEQ_LEN), (BATCH_SIZE, NUM_KEY_VALUE_HEADS, SEQ_LEN, HEAD_DIM)]),
        ("attention_output_projection", AttentionOutputProjection, [(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)]),
        ("mlp_gate_up_projection", MLPGateUpProjection, [(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)]),
        ("mlp_activation", MLPActivation, [(BATCH_SIZE, SEQ_LEN, 2 * INTERMEDIATE_SIZE)]),
        ("mlp_down_projection", MLPDownProjection, [(BATCH_SIZE, SEQ_LEN, INTERMEDIATE_SIZE)]),
        ("rms_norm", RMSNorm, [(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)]),
    ]
    
    print(f"Starting Step 1: E2E Model Tracing and Global Tuning")
    print(f"Total Global Trial Budget: {TOTAL_TRIALS}")
    
    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    
    # Phase A: Build a massive global list of every task from all 9 operators
    all_global_tasks = []
    for i, (name, cls, shapes) in enumerate(operators, 1):
        extracted_tasks = extract_tasks_for_operator(cls, shapes, name, target)
        all_global_tasks.extend(extracted_tasks)
        
    print("\n" + "="*80)
    print(f"Found {len(all_global_tasks)} distinct tuning tasks across all operators.")
    print("Passing control to TVM's Global TaskScheduler...")
    print("="*80)

    # Phase B: Tune the entire pool globally.
    if not IS_IN_CI:
        work_dir = "./tuning_logs_" + MODEL_NAME
        
        # 1. Convert the raw extracted tasks into MetaSchedule TuneContexts and calculate FLOP weights
        tasks, task_weights = ms.relax_integration.extracted_tasks_to_tune_contexts(
            extracted_tasks=all_global_tasks,
            work_dir=work_dir,
        )
        
        # 2. Feed them into the Global Task Scheduler
        database = ms.tune_tasks(
            tasks=tasks,
            task_weights=task_weights,
            work_dir=work_dir,
            max_trials_global=TOTAL_TRIALS,
            num_trials_per_iter=64,
        )
        
        print(f"\nTuning completed. Log saved to {work_dir}")

if __name__ == "__main__":
    main()
