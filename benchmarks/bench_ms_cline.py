#!/usr/bin/env python3
# bench_ms_cline.py
# End-to-end comparison: measure PyTorch and TVM (Relax + MetaSchedule) performance on ResNet.
# Usage:
#   python3 bench_ms_cline.py --model resnet18 --batch-size 1 --target "cuda" --device "cuda" --trials 2000

import argparse
import time
import numpy as np
import tvm
from tvm import relax

import torch
from torch.export import export as torch_export
from torchvision.models.resnet import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

def parse_args():
    p = argparse.ArgumentParser(description="Compare PyTorch vs TVM(RELAX+MetaSchedule) inference latency.")
    p.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--target", type=str, default="cuda", help='Compilation target, e.g., "cuda" or "llvm"')
    p.add_argument("--device", type=str, default="cuda", help='Runtime device, "cuda" or "cpu"')
    p.add_argument("--trials", type=int, default=2000, help="Total tuning trials in pipeline (0 to skip tuning)")
    p.add_argument("--number", type=int, default=10, help="Number of runs per repeat")
    p.add_argument("--repeat", type=int, default=3, help="Repeat count for benchmarking")
    p.add_argument("--warmup", type=int, default=3, help="Warmup runs")
    return p.parse_args()

def measure_pytorch(model, input_tensor, repeat, number, warmup):
    device = input_tensor.device
    is_cuda = device.type == "cuda"
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(input_tensor)
            if is_cuda:
                torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for r in range(repeat):
            start = time.perf_counter()
            for _ in range(number):
                model(input_tensor)
            if is_cuda:
                torch.cuda.synchronize()
            end = time.perf_counter()
            per_call_ms = (end - start) * 1000.0 / number
            times.append(per_call_ms)
    return np.array(times)

def measure_tvm_vm(vm, in_array, tvm_params, func_name, repeat, number, warmup):
    # Warmup
    for _ in range(warmup):
        vm[func_name](in_array, *tvm_params)
    times = []
    for r in range(repeat):
        start = time.perf_counter()
        for _ in range(number):
            vm[func_name](in_array, *tvm_params)
        end = time.perf_counter()
        per_call_ms = (end - start) * 1000.0 / number
        times.append(per_call_ms)
    return np.array(times)

def main():
    args = parse_args()
    batch = args.batch_size
    input_shape = (batch, 3, args.height, args.width)
    model_name = args.model

    # Prepare PyTorch model
    print("Preparing PyTorch model...")
    if model_name == "resnet18":
        torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    else:
        torch_model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()

    # Prepare PyTorch input and device
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    pt_device = torch.device("cuda" if use_cuda else "cpu")
    torch_model = torch_model.to(pt_device)
    input_torch = torch.randn(*input_shape, dtype=torch.float32, device=pt_device)

    # Measure PyTorch baseline
    print("Measuring PyTorch baseline...")
    pt_times = measure_pytorch(torch_model, input_torch, args.repeat, args.number, args.warmup)
    pt_mean = float(pt_times.mean())
    pt_std = float(pt_times.std())
    print(f"PyTorch latency (ms): mean={pt_mean:.3f}, std={pt_std:.3f} (per run)")

    # Export and convert to Relax IRModule
    print("Exporting PyTorch program and converting to Relax IRModule...")
    example_args = (torch.randn(*input_shape, dtype=torch.float32),)
    exported_program = torch_export(torch_model.eval(), example_args)
    from tvm.relax.frontend.torch import from_exported_program
    mod = mod = from_exported_program(exported_program, keep_params_as_input=True)
    mod, params = relax.frontend.detach_params(mod)

    # Tuning pipeline (MetaSchedule) via Relax pipeline
    target = tvm.target.Target(args.target)
    if args.trials > 0:
        print(f"Running Relax tuning pipeline (trials={args.trials}) ...")
        mod = relax.get_pipeline("static_shape_tuning", target=target, total_trials=args.trials)(mod)
        print("Tuning pipeline finished.")
    else:
        print("Skipping tuning pipeline (trials=0).")

    # Compile
    print("Compiling optimized module...")
    ex = tvm.compile(mod, target=str(target))
    dev_type = args.device if args.device in ("cuda", "cpu") else "cpu"
    dev = tvm.device(dev_type, 0)
    vm = relax.VirtualMachine(ex, dev)

    # Prepare input and params on device
    in_np = np.random.rand(*input_shape).astype("float32")
    in_tvm = tvm.nd.array(in_np, dev)
    # params can be mapping; try to get list under "main"
    param_list = params.get("main", params) if isinstance(params, dict) else params
    if isinstance(param_list, dict):
        param_list = param_list.get("main", [])
    tvm_params = [tvm.nd.array(p, dev) for p in param_list]

    # Measure TVM (VM)
    print("Measuring TVM (Relax VM) ...")
    tvm_times = measure_tvm_vm(vm, in_tvm, tvm_params, "main", args.repeat, args.number, args.warmup)
    tvm_mean = float(tvm_times.mean())
    tvm_std = float(tvm_times.std())
    print(f"TVM (Relax VM) latency (ms): mean={tvm_mean:.3f}, std={tvm_std:.3f} (per run)")

    # Print comparison
    print("\nSummary (ms per run):")
    print(f"PyTorch: mean={pt_mean:.3f}  std={pt_std:.3f}")
    print(f"TVM   : mean={tvm_mean:.3f}  std={tvm_std:.3f}")
    ratio = pt_mean / tvm_mean if tvm_mean > 0 else float("inf")
    print(f"Speedup (PyTorch / TVM): {ratio:.3f}x")

if __name__ == "__main__":
    main()
