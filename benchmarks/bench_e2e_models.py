# -*- coding: utf-8 -*-
import os, time, argparse, json, math, statistics as stats
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# ---------------------------
# Model list for benchmarking (can be replaced with official large models as needed)
# ---------------------------
MODEL_ZOO = {
    "bert-tiny":     "prajjwal1/bert-tiny",                       # or hf-internal-testing/tiny-random-BertModel
    "bert-base":     "bert-base-uncased",
    "gpt2":          "gpt2",
    "llama-tiny":    "hf-internal-testing/tiny-random-LlamaModel",
    "opt-125m":      "facebook/opt-125m",
    "mistral-tiny":  "hf-internal-testing/tiny-random-MistralModel",
}

# ---------------------------
# Wrappers: Wrap HF models to only accept Tensor inputs for forward, convenient for torch.export
# ---------------------------
class EncoderWrapper(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.core(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return out.last_hidden_state  # [B, S, H]

class DecoderWrapper(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.core(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        # For AutoModel (not *ForCausalLM), also return last_hidden_state
        return out.last_hidden_state  # [B, S, H]

# ---------------------------
# Input construction
# ---------------------------
def make_inputs(cfg, bs: int, seq_len: int, device: str = "cuda"):
    vocab = int(getattr(cfg, "vocab_size", 50257))
    input_ids = torch.randint(0, vocab, (bs, seq_len), dtype=torch.long, device=device)
    attention = torch.ones((bs, seq_len), dtype=torch.long, device=device)
    return input_ids, attention

# ---------------------------
# Timing utility
# ---------------------------
def bench_call(fn, warmup=10, iters=50, sync=lambda: None) -> float:
    # Returns milliseconds
    for _ in range(warmup):
        _ = fn()
        sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fn()
        sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return stats.median(times)

# ---------------------------
# PyTorch / TorchInductor path
# ---------------------------
def run_pytorch(model, inputs):
    input_ids, attention = inputs
    with torch.inference_mode():
        return bench_call(lambda: model(input_ids, attention), sync=torch.cuda.synchronize)

def run_inductor(model, inputs):
    compiled = torch.compile(model, backend="inductor", mode="max-autotune")
    input_ids, attention = inputs
    with torch.inference_mode():
        return bench_call(lambda: compiled(input_ids, attention), sync=torch.cuda.synchronize)

# ---------------------------
# TVM Unity + MetaSchedule path
# ---------------------------
def run_tvm_ms(wrapper_cpu: nn.Module, example_inputs_cpu: Tuple[torch.Tensor, torch.Tensor],
               target: str, total_trials: int):
    # 1) export -> IRModule
    from torch.export import export
    with torch.no_grad():
        exported = export(wrapper_cpu.eval(), example_inputs_cpu)
        mod = from_exported_program(exported, keep_params_as_input=True)
    mod, params = relax.frontend.detach_params(mod)

    # 2) Static shape tuning (you can change the pipeline to your custom one)
    pipeline = relax.get_pipeline("static_shape_tuning", target=tvm.target.Target(target),
                                  total_trials=total_trials)
    mod = pipeline(mod)

    # 3) Compile and execute
    rt_target = "cuda" if "nvidia" in target or "cuda" in target else target
    ex = tvm.compile(mod, target=rt_target)
    dev = tvm.device(rt_target, 0)
    vm = relax.VirtualMachine(ex, dev)

    # 4) Prepare inputs/params to GPU
    #   - In IR, main's parameter order: input tensors first, then params list
    ex_inp0 = example_inputs_cpu[0].contiguous().numpy()
    ex_inp1 = example_inputs_cpu[1].contiguous().numpy()
    tvm_inp0 = tvm.nd.array(ex_inp0, dev)
    tvm_inp1 = tvm.nd.array(ex_inp1, dev)
    tvm_params = [tvm.nd.array(p, dev) for p in params["main"]]

    def _call():
        return vm["main"](tvm_inp0, tvm_inp1, *tvm_params)

    # Note: tvm.cuda(0).sync() for synchronization
    latency_ms = bench_call(_call, sync=lambda: tvm.cuda(0).sync())
    return latency_ms

# ---------------------------
# Main workflow
# ---------------------------
@dataclass
class TaskSpec:
    family: str
    hf_id: str
    is_encoder: bool

def build_task_list() -> List[TaskSpec]:
    return [
        TaskSpec("bert-tiny",    MODEL_ZOO["bert-tiny"],    True),
        TaskSpec("bert-base",    MODEL_ZOO["bert-base"],    True),
        TaskSpec("gpt2",         MODEL_ZOO["gpt2"],         False),
        TaskSpec("llama-tiny",   MODEL_ZOO["llama-tiny"],   False),
        TaskSpec("opt-125m",     MODEL_ZOO["opt-125m"],     False),
        TaskSpec("mistral-tiny", MODEL_ZOO["mistral-tiny"], False),
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[1,4])
    parser.add_argument("--seq-len", type=int, default=512)     # BERT can be set to 128~384; LLM 512/1024/2048
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32","fp16"])
    parser.add_argument("--modes", type=str, nargs="+",
                        default=["pytorch","torch_compile","tvm_ms"])
    parser.add_argument("--tvm-target", type=str, default="nvidia/geforce-rtx-4090")
    parser.add_argument("--tvm-trials", type=int, default=2000) # Can be increased to 20000
    parser.add_argument("--out", type=str, default="results_e2e.json")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = "cuda"
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    results: Dict[str, Dict[str, float]] = {}
    tasks = build_task_list()

    for task in tasks:
        print(f"\n=== {task.family} ===")
        cfg = AutoConfig.from_pretrained(task.hf_id, trust_remote_code=True)
        # Some models default to use_cache=True, which affects export; turn it off
        if hasattr(cfg, "use_cache"):
            cfg.use_cache = False

        core = AutoModel.from_pretrained(task.hf_id, torch_dtype=dtype, low_cpu_mem_usage=True,
                                         trust_remote_code=True)
        core.eval().to(device)

        wrapper = EncoderWrapper(core) if task.is_encoder else DecoderWrapper(core)
        wrapper = wrapper.eval().to(device)

        results[task.family] = {}

        for bs in args.batches:
            print(f"  - bs={bs}")
            # Build input
            in_gpu = make_inputs(cfg, bs, args.seq_len, device=device)

            # 1) PyTorch eager
            if "pytorch" in args.modes:
                t_ms = run_pytorch(wrapper, in_gpu)
                results[task.family][f"pytorch_bs{bs}"] = t_ms
                print(f"    pytorch:        {t_ms:.2f} ms")

            # 2) TorchInductor(+Triton)
            if "torch_compile" in args.modes:
                t_ms = run_inductor(wrapper, in_gpu)
                results[task.family][f"inductor_bs{bs}"] = t_ms
                print(f"    torch_compile:  {t_ms:.2f} ms")

            # 3) TVM MetaSchedule
            if "tvm_ms" in args.modes:
                # torch.export is currently most stable when exporting on CPU
                wrapper_cpu = wrapper.to("cpu")
                in_cpu = tuple(x.to("cpu") for x in in_gpu)
                with torch.no_grad():
                    t_ms = run_tvm_ms(wrapper_cpu, in_cpu, args.tvm_target, args.tvm_trials)
                results[task.family][f"tvm_ms_bs{bs}"] = t_ms
                print(f"    tvm_ms:         {t_ms:.2f} ms")

            torch.cuda.empty_cache()

    # Summary + normalization (smaller is better: take PyTorch latency / X latency)
    table = []
    for fam, kv in results.items():
        for bs in args.batches:
            base = kv.get(f"pytorch_bs{bs}")
            row = {
                "model": fam, "bs": bs,
                "pytorch_ms": kv.get(f"pytorch_bs{bs}", math.nan),
                "inductor_ms": kv.get(f"inductor_bs{bs}", math.nan),
                "tvm_ms_ms": kv.get(f"tvm_ms_bs{bs}", math.nan),
            }
            if base and base > 0:
                row["norm_pytorch"]  = 1.0
                row["norm_inductor"] = base / row["inductor_ms"] if row["inductor_ms"] else math.nan
                row["norm_tvm_ms"]   = base / row["tvm_ms_ms"]   if row["tvm_ms_ms"]   else math.nan
            table.append(row)

    with open(args.out, "w") as f:
        json.dump({"results": results, "summary": table}, f, indent=2)
    print(f"\nSaved to {args.out}")
    print("\n==== Summary (normalized, higher is better) ====")
    for r in table:
        if r["bs"] in (1,4):
            print(f'{r["model"]:12s} bs={r["bs"]} | '
                  f'PyTorch {r["norm_pytorch"]:.2f}  '
                  f'Inductor {r.get("norm_inductor", float("nan")):.2f}  '
                  f'TVM-MS {r.get("norm_tvm_ms", float("nan")):.2f}')
if __name__ == "__main__":
    main()
