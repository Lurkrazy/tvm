#!/usr/bin/env python3
# Minimal PyTorch script to measure a single ResNet-50 layer or end-to-end ResNet-50.
# Modified to align workload with docs/how_to/tutorials/e2e_opt_model.py (use float32 end-to-end).
# Usage examples:
#   Single layer (tensor-core FP16): python3 bench_resnet50_tensorcore_pt.py --mode layer --layer conv1 --runs 50 --warmup 5 --dtype float16
#   End-to-end aligned (float32):    python3 bench_resnet50_tensorcore_pt.py --mode e2e   --runs 50 --warmup 5 --dtype float32
#
# The script prints mean/median/std latency and sample timings.

import argparse
import time
import statistics
import torch
import numpy as np
from torchvision.models.resnet import resnet50, ResNet50_Weights

def get_submodule(module, path):
    """Get submodule by dotted path like 'layer1.0.conv1' or 'conv1'."""
    parts = path.split(".")
    m = module
    for p in parts:
        if p.isdigit():
            m = m[int(p)]
        else:
            m = getattr(m, p)
    return m

def measure_fn(fn, input_tensor, runs, warmup, sync=True):
    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn(input_tensor)
            if sync and input_tensor.is_cuda:
                torch.cuda.synchronize()
    # timed runs
    times_ms = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = fn(input_tensor)
            if sync and input_tensor.is_cuda:
                torch.cuda.synchronize()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)
    return times_ms

def main():
    p = argparse.ArgumentParser(description="ResNet-50 timing (single-layer or end-to-end). Align e2e with e2e_opt_model.py by using float32.")
    p.add_argument("--mode", choices=["layer", "e2e"], default="layer", help="Measure a layer or end-to-end model")
    p.add_argument("--layer", type=str, default="conv1", help="Layer path to test when mode=layer (e.g. conv1 or layer1.0.conv1)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dtype", choices=["float32", "float16"], default=None, help="Input dtype. If not set: layer->float16, e2e->float32 (to align with e2e_opt_model.py)")
    p.add_argument("--seed", type=int, default=0, help="Random seed to make inputs reproducible")
    args = p.parse_args()

    # reproducible input
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    if args.mode == "layer" and not use_cuda:
        raise SystemExit("CUDA is recommended for layer tensor-core testing. No CUDA device found.")

    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        props = torch.cuda.get_device_properties(device)
        print(f"CUDA device: {props.name}, compute capability: {props.major}.{props.minor}")
        print("Recommend compute capability >= 7.0 for Tensor Cores (Volta/Turing/Ampere).")
    else:
        print("CUDA not available; running on CPU.")

    # Determine dtype
    if args.dtype is not None:
        dtype = torch.float16 if args.dtype == "float16" else torch.float32
    else:
        # default: layer -> float16 (tensor-core), e2e -> float32 (align with e2e_opt_model.py)
        dtype = torch.float16 if args.mode == "layer" else torch.float32

    print(f"Selected dtype for measurement: {dtype}")

    # Build model
    model = resnet50(weights=ResNet50_Weights.DEFAULT).eval().to(device)

    if args.mode == "layer":
        sub = get_submodule(model, args.layer)
        try:
            sub.half() if dtype == torch.float16 else sub.float()
        except Exception:
            print("Warning: failed to cast submodule dtype; continuing with original dtype.")
        input_tensor = torch.randn(args.batch_size, 3, args.height, args.width, device=device, dtype=dtype)
        fn = lambda x: sub(x)
    else:
        # e2e: align with e2e_opt_model.py which uses float32 inputs/params before TVM tuning
        if dtype == torch.float16:
            model.half()
        else:
            model.float()
        input_tensor = torch.randn(args.batch_size, 3, args.height, args.width, device=device, dtype=dtype)
        fn = lambda x: model(x)

    # Turn on cuDNN autotuner for better performance on many configs
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    print(f"Measuring mode={args.mode}, layer={args.layer if args.mode=='layer' else 'full model'}, input shape={tuple(input_tensor.shape)}, seed={args.seed}")
    times_ms = measure_fn(fn, input_tensor, runs=args.runs, warmup=args.warmup, sync=use_cuda)

    mean_ms = statistics.mean(times_ms)
    stdev_ms = statistics.pstdev(times_ms)
    median_ms = statistics.median(times_ms)

    print(f"Runs: {args.runs}, Warmup: {args.warmup}")
    print(f"Latency (ms) per run: mean={mean_ms:.3f}, median={median_ms:.3f}, std={stdev_ms:.3f}")
    print("Sample timings (ms):", ", ".join(f"{t:.3f}" for t in times_ms[:10]))

    # Save the input used (for reproducible cross-run comparison)
    # This lets you feed the exact same input into TVM conversion/benchmark if needed.
    in_np = input_tensor.cpu().numpy()
    np.save("bench_resnet50_input.npy", in_np)
    print("Saved input to bench_resnet50_input.npy for cross-framework alignment.")

if __name__ == "__main__":
    main()
