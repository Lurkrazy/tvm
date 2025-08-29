import os
import numpy as np
import torch
from torch.export import export
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
import logging
import time

# Configure logging for MetaSchedule
logging.basicConfig(level=logging.INFO)
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)

def tune_and_benchmark_resnet(
    model_name: str = "resnet-18",
    batch_size: int = 1,
    image_shape: tuple = (3, 224, 224),
    dtype: str = "float32",
    target_str: str = "cuda",
    work_dir: str = "resnet_meta_schedule_tuning_logs",
    total_trials: int = 8000,
):
    """
    Tunes a ResNet model using MetaSchedule and benchmarks its performance.

    Parameters
    ----------
    model_name : str
        Name of the ResNet model (e.g., "resnet-18", "resnet-50").
    batch_size : int
        Batch size for inference.
    image_shape : tuple
        Shape of the input image (C, H, W).
    dtype : str
        Data type of the model (e.g., "float32").
    target_str : str
        TVM target string (e.g., "cuda", "llvm").
    work_dir : str
        Directory to store MetaSchedule tuning logs and database.
    total_trials : int
        Total number of trials for MetaSchedule tuning.
    """

    print(f"--- Starting tuning and benchmarking for {model_name} on {target_str} ---")

    # 1. Load the ResNet Model from PyTorch
    data_shape = (batch_size,) + image_shape
    
    if model_name == "resnet-18":
        torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        num_layers = 18
    elif model_name == "resnet-50":
        torch_model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
        num_layers = 50
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    print(f"Loaded PyTorch {model_name} model")

    # 2. Convert to TVM IRModule using Relax frontend
    example_args = (torch.randn(*data_shape, dtype=getattr(torch, dtype)),)
    
    print("Converting PyTorch model to TVM IRModule...")
    with torch.no_grad():
        exported_program = export(torch_model, example_args)
        mod = from_exported_program(exported_program, keep_params_as_input=True)

    # Detach parameters from the module
    mod, params = relax.frontend.detach_params(mod)
    print("Model converted to TVM IRModule")

    # 3. Set up target and tuning
    target = tvm.target.Target(target_str)
    print(f"Target: {target}")

    # 4. Optimize with MetaSchedule using Relax pipeline
    print(f"Starting MetaSchedule tuning with {total_trials} trials...")
    print(f"Tuning logs will be stored in: {work_dir}")
    
    # Use the static_shape_tuning pipeline which includes MetaSchedule
    optimized_mod = relax.get_pipeline(
        "static_shape_tuning", 
        target=target, 
        total_trials=total_trials,
        work_dir=work_dir
    )(mod)
    
    print("MetaSchedule tuning finished.")

    # 5. Build the optimized model
    print("Building optimized model...")
    ex = tvm.compile(optimized_mod, target=target_str)
    print("Model compilation finished.")

    # 6. Evaluate the Compiled Model
    print("--- Benchmarking the compiled model ---")
    
    # Set up device and virtual machine
    if "cuda" in target_str:
        dev = tvm.device("cuda", 0)
    else:
        dev = tvm.device("cpu", 0)
        
    vm = relax.VirtualMachine(ex, dev)
    
    # Prepare input data and parameters on device
    input_data = np.random.rand(*data_shape).astype(dtype)
    gpu_data = tvm.nd.array(input_data, dev)
    gpu_params = [tvm.nd.array(p, dev) for p in params["main"]]

    # Warm-up runs
    print("Performing warm-up runs...")
    for _ in range(5):
        _ = vm["main"](gpu_data, *gpu_params)

    # Measure execution time
    print("Measuring inference time...")
    num_runs = 100
    start_time = time.time()
    
    for _ in range(num_runs):
        output = vm["main"](gpu_data, *gpu_params)
    
    end_time = time.time()
    
    # Synchronize device if CUDA
    if "cuda" in target_str:
        dev.sync()
    
    total_time = end_time - start_time
    mean_inference_time_ms = (total_time / num_runs) * 1000
    
    print(f"Average inference time: {mean_inference_time_ms:.3f} ms")
    print(f"Output shape: {output.shape}")

    print(f"--- Tuning and benchmarking for {model_name} on {target_str} completed ---")
    return mean_inference_time_ms

if __name__ == "__main__":
    # Example usage: Tune and benchmark ResNet-18 on CUDA
    cuda_time = tune_and_benchmark_resnet(
        model_name="resnet-18",
        target_str="nvidia/geforce-rtx-4090",
        work_dir="resnet18_cuda_logs",
        total_trials=8000,  # Increase for better performance
    )
    print(f"\nResNet-18 on CUDA average inference time: {cuda_time:.3f} ms")

    # Example usage: Tune and benchmark ResNet-18 on CPU
    # cpu_time = tune_and_benchmark_resnet(
    #     model_name="resnet-18",
    #     target_str="llvm -mcpu=skylake-avx512",
    #     work_dir="resnet18_cpu_logs",
    #     total_trials=4000,
    # )
    # print(f"\nResNet-18 on CPU average inference time: {cpu_time:.3f} ms")