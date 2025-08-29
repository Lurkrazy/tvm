import argparse
import logging
import os

import numpy as np
import torch
from torch.export import export
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

import tvm
from tvm import meta_schedule as ms
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# Configure logging for MetaSchedule
logging.basicConfig(level=logging.INFO)
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


def get_torch_model_for_relax(model_name: str, batch_size: int):
    """Load a PyTorch ResNet model and convert it to a Relax IRModule."""
    if model_name == "resnet-18":
        torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    elif model_name == "resnet-50":
        torch_model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    example_args = (torch.randn(*data_shape, dtype=torch.float32),)

    with torch.no_grad():
        exported_program = export(torch_model, example_args)
        mod = from_exported_program(exported_program, keep_params_as_input=True)
        
    mod, params = relax.frontend.detach_params(mod)
    return mod, params, data_shape, "float32"


def tune_and_benchmark_resnet(
    model_name: str,
    batch_size: int,
    target_str: str,
    work_dir: str,
    max_trials_global: int,
):
    """
    Tunes a ResNet model using MetaSchedule and benchmarks its performance.
    """

    print(f"--- Starting tuning and benchmarking for {model_name} on {target_str} ---")

    # 1. Load the ResNet Model into Relax IRModule
    print("\n--- 1. Loading PyTorch model and converting to Relax IRModule ---")
    mod, params, data_shape, dtype = get_torch_model_for_relax(model_name, batch_size)
    target = tvm.target.Target(target_str)

    # 2. Tune the model using the high-level static_shape_tuning pipeline
    # This pipeline handles task extraction, tuning, and applying the best schedules.
    print(f"\n--- 2. Starting MetaSchedule tuning via pipeline in '{work_dir}' ---")
    mod = relax.get_pipeline(
        "static_shape_tuning",
        target=target,
        total_trials=max_trials_global,
        work_dir=work_dir,
    )(mod)

    # 3. Compile the tuned model
    print("\n--- 3. Compiling tuned model ---")
    ex = tvm.compile(mod, target=target)

    # 4. Benchmark the compiled module using the VirtualMachine
    print("\n--- 4. Benchmarking performance ---")
    dev = tvm.device(str(target), 0)
    vm = relax.VirtualMachine(ex, dev)

    # Prepare inputs and parameters on the target device
    data_tvm = tvm.nd.array(np.random.uniform(size=data_shape).astype(dtype), dev)
    # The params from `detach_params` is a dict where keys are function names.
    # We need the params for the "main" function.
    device_params = [tvm.nd.array(p, dev) for p in params["main"]]

    # Evaluate
    print("Evaluating inference time cost...")
    ftimer = vm.time_evaluator(
        "main", dev, number=1, repeat=100, args=[data_tvm, *device_params]
    )
    prof_res = ftimer()
    mean_ms = prof_res.mean * 1000
    std_ms = prof_res.std * 1000
    print(
        f"Mean inference time (std dev): {mean_ms:.2f} ms ({std_ms:.2f} ms)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet-18", help="The ResNet model to use.")
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size.")
    parser.add_argument(
        "--target",
        default="cuda",
        help="The TVM target string.",
    )
    parser.add_argument("--work-dir", default="resnet_tuning_logs", help="The directory for logs.")
    parser.add_argument("--trials", type=int, default=2000, help="The number of tuning trials.")
    args = parser.parse_args()

    tune_and_benchmark_resnet(
        model_name=args.model,
        batch_size=args.batch_size,
        target_str=args.target,
        work_dir=args.work_dir,
        max_trials_global=args.trials,
    )


if __name__ == "__main__":
    main()
