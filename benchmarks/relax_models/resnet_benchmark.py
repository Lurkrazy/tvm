#!/usr/bin/env python3
"""
ResNet End-to-End Benchmark with TVM Relax Auto-Tuning

This script demonstrates TVM Relax's auto-tuning capabilities on ResNet models.
It compares performance with other inference frameworks and showcases various
optimization techniques including tensor core utilization.

Usage:
    python resnet_benchmark.py --model resnet18 --batch-size 1 --tune --target cuda
    python resnet_benchmark.py --model resnet50 --batch-size 32 --compare-frameworks
"""

import argparse
import time
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch.fx import symbolic_trace

import tvm
from tvm import relax, meta_schedule as ms
from tvm.relax.frontend.torch import from_fx
from tvm.relax import pipeline


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResNetBenchmark:
    """ResNet benchmark class for TVM Relax auto-tuning evaluation."""
    
    def __init__(self, model_name: str, batch_size: int, input_shape: Tuple[int, ...], 
                 target: str = "cuda", enable_tuning: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.target = tvm.target.Target(target)
        self.enable_tuning = enable_tuning
        
        # Model setup
        self.torch_model = self._load_torch_model()
        self.torch_model.eval()
        
        # Input data
        self.input_data = torch.randn(batch_size, *input_shape[1:])
        
        # TVM setup
        self.relax_mod = None
        self.relax_exec = None
        self.tuning_database = None
        
        logger.info(f"Initialized {model_name} benchmark: batch_size={batch_size}, target={target}")
    
    def _load_torch_model(self) -> torch.nn.Module:
        """Load PyTorch model."""
        model_map = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
        }
        
        if self.model_name not in model_map:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Load pre-trained model
        model = model_map[self.model_name](pretrained=True)
        return model
    
    def convert_to_relax(self) -> tvm.IRModule:
        """Convert PyTorch model to TVM Relax IR."""
        logger.info("Converting PyTorch model to TVM Relax IR...")
        
        with torch.no_grad():
            # Trace the model
            traced_model = symbolic_trace(self.torch_model)
            
            # Convert to Relax
            input_info = [("input", self.input_shape, "float32")]
            mod = from_fx(traced_model, input_info)
            
        self.relax_mod = mod
        logger.info("Model conversion completed")
        return mod
    
    def apply_optimizations(self, mod: tvm.IRModule) -> tvm.IRModule:
        """Apply TVM Relax optimization passes."""
        logger.info("Applying optimization passes...")
        
        # Standard optimization pipeline
        seq = tvm.transform.Sequential([
            # Graph-level optimizations
            relax.transform.DecomposeOpsForInference(),
            relax.transform.CanonicalizeBindings(),
            relax.transform.EliminateCommonSubexpr(),
            relax.transform.CombineParallelMatmul(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
            
            # Layout optimizations for tensor cores
            relax.transform.ConvertLayout({"relax.nn.conv2d": ["NHWC", "auto"]}),
            relax.transform.FoldConstant(),
            
            # Prepare for tuning
            relax.transform.LegalizeOps(),
        ])
        
        optimized_mod = seq(mod)
        logger.info("Optimization passes completed")
        return optimized_mod
    
    def auto_tune_model(self, mod: tvm.IRModule, work_dir: str = "./tune_logs") -> tvm.IRModule:
        """Auto-tune the model using TVM meta-schedule."""
        if not self.enable_tuning:
            logger.info("Auto-tuning disabled, skipping...")
            return mod
            
        logger.info("Starting auto-tuning with meta-schedule...")
        
        # Create work directory
        os.makedirs(work_dir, exist_ok=True)
        
        # Configure tuning parameters
        num_trials = 1000  # Adjust based on available time
        if "resnet18" in self.model_name or "resnet34" in self.model_name:
            num_trials = 500  # Smaller models need fewer trials
        elif "resnet152" in self.model_name:
            num_trials = 2000  # Larger models benefit from more trials
        
        # Setup meta-schedule database
        database = ms.tir_integration.database.JSONDatabase(
            path_workload=os.path.join(work_dir, "database_workload.json"),
            path_tuning_record=os.path.join(work_dir, "database_tuning_record.json"),
        )
        
        # Auto-tuning pipeline with tensor core support
        with ms.Profiler() as profiler:
            tuned_mod = pipeline.static_shape_tuning_pipeline(
                mod=mod,
                target=self.target,
                work_dir=work_dir,
                max_trials_global=num_trials,
                
                # Advanced tuning options
                num_trials_per_iter=32,
                max_trials_per_task=128,
                
                # Enable tensor core optimizations for CUDA
                schedule_rules=ms.default_config.schedule_rules(target=self.target) if "cuda" in str(self.target) else None,
                postprocs=ms.default_config.postproc(target=self.target),
                
                # Cost model configuration
                strategy="evolutionary",
                num_tuning_cores=os.cpu_count(),
            )
        
        # Save tuning database
        self.tuning_database = database
        
        tuning_time = profiler.get()["total_time"]
        logger.info(f"Auto-tuning completed in {tuning_time:.2f} seconds")
        
        return tuned_mod
    
    def build_model(self, mod: tvm.IRModule) -> relax.VirtualMachine:
        """Build the optimized model."""
        logger.info("Building optimized model...")
        
        # Build with optimizations
        exec = relax.build(mod, target=self.target)
        vm = relax.VirtualMachine(exec, tvm.device(str(self.target).split()[0], 0))
        
        self.relax_exec = exec
        logger.info("Model build completed")
        return vm
    
    def benchmark_tvm_relax(self, num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
        """Benchmark TVM Relax performance."""
        if self.relax_exec is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        logger.info(f"Benchmarking TVM Relax ({num_runs} runs)...")
        
        # Create VM and prepare input
        vm = relax.VirtualMachine(self.relax_exec, tvm.device(str(self.target).split()[0], 0))
        tvm_input = tvm.nd.array(self.input_data.numpy())
        
        # Warmup runs
        for _ in range(warmup_runs):
            vm["main"](tvm_input)
        
        # Synchronize before timing
        tvm.runtime.Device(str(self.target).split()[0], 0).sync()
        
        # Benchmark runs
        start_time = time.time()
        for _ in range(num_runs):
            output = vm["main"](tvm_input)
        
        # Synchronize and measure
        tvm.runtime.Device(str(self.target).split()[0], 0).sync()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        throughput = self.batch_size / (avg_time / 1000)  # samples/sec
        
        results = {
            "avg_latency_ms": avg_time,
            "throughput_samples_per_sec": throughput,
            "total_runs": num_runs,
            "batch_size": self.batch_size
        }
        
        logger.info(f"TVM Relax Results: {avg_time:.2f}ms avg latency, {throughput:.1f} samples/sec")
        return results
    
    def benchmark_pytorch(self, num_runs: int = 100, warmup_runs: int = 10, 
                         use_cuda: bool = True) -> Dict[str, float]:
        """Benchmark PyTorch performance for comparison."""
        logger.info(f"Benchmarking PyTorch ({num_runs} runs)...")
        
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        model = self.torch_model.to(device)
        input_tensor = self.input_data.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        if use_cuda:
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(input_tensor)
        
        if use_cuda:
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        throughput = self.batch_size / (avg_time / 1000)  # samples/sec
        
        results = {
            "avg_latency_ms": avg_time,
            "throughput_samples_per_sec": throughput,
            "total_runs": num_runs,
            "batch_size": self.batch_size,
            "device": str(device)
        }
        
        logger.info(f"PyTorch Results: {avg_time:.2f}ms avg latency, {throughput:.1f} samples/sec")
        return results
    
    def run_full_benchmark(self, work_dir: str = "./tune_logs") -> Dict:
        """Run complete benchmark pipeline."""
        logger.info("Starting full benchmark pipeline...")
        
        results = {
            "model": self.model_name,
            "batch_size": self.batch_size,
            "target": str(self.target),
            "tuning_enabled": self.enable_tuning,
        }
        
        try:
            # Convert model
            mod = self.convert_to_relax()
            
            # Apply optimizations
            optimized_mod = self.apply_optimizations(mod)
            
            # Auto-tune if enabled
            if self.enable_tuning:
                tuned_mod = self.auto_tune_model(optimized_mod, work_dir)
            else:
                tuned_mod = optimized_mod
            
            # Build model
            vm = self.build_model(tuned_mod)
            
            # Benchmark TVM Relax
            tvm_results = self.benchmark_tvm_relax()
            results["tvm_relax"] = tvm_results
            
            # Benchmark PyTorch for comparison
            pytorch_results = self.benchmark_pytorch()
            results["pytorch"] = pytorch_results
            
            # Calculate speedup
            speedup = pytorch_results["avg_latency_ms"] / tvm_results["avg_latency_ms"]
            results["speedup"] = speedup
            
            logger.info(f"Benchmark completed. TVM Relax speedup: {speedup:.2f}x")
            
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            results["error"] = str(e)
            raise
        
        return results


def benchmark_with_onnxruntime(model_name: str, input_data: np.ndarray, 
                              num_runs: int = 100) -> Optional[Dict[str, float]]:
    """Benchmark with ONNX Runtime for comparison."""
    try:
        import onnxruntime as ort
        import torch.onnx
        
        logger.info("Benchmarking ONNX Runtime...")
        
        # Export to ONNX
        torch_model = getattr(models, model_name)(pretrained=True)
        torch_model.eval()
        
        dummy_input = torch.randn_like(torch.from_numpy(input_data))
        onnx_path = f"/tmp/{model_name}.onnx"
        
        with torch.no_grad():
            torch.onnx.export(
                torch_model, dummy_input, onnx_path,
                export_params=True, opset_version=11,
                input_names=['input'], output_names=['output']
            )
        
        # Load with ONNX Runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Warmup
        for _ in range(10):
            session.run(None, {'input': input_data})
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            session.run(None, {'input': input_data})
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000
        throughput = input_data.shape[0] / (avg_time / 1000)
        
        # Cleanup
        os.remove(onnx_path)
        
        results = {
            "avg_latency_ms": avg_time,
            "throughput_samples_per_sec": throughput,
            "provider": session.get_providers()[0]
        }
        
        logger.info(f"ONNX Runtime Results: {avg_time:.2f}ms avg latency")
        return results
        
    except ImportError:
        logger.warning("ONNX Runtime not available for comparison")
        return None
    except Exception as e:
        logger.error(f"ONNX Runtime benchmark failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="ResNet TVM Relax Auto-Tuning Benchmark")
    parser.add_argument("--model", default="resnet50", 
                       choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                       help="ResNet model variant")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--target", default="cuda", help="TVM target (cuda, cpu, etc.)")
    parser.add_argument("--tune", action="store_true", help="Enable auto-tuning")
    parser.add_argument("--compare-frameworks", action="store_true", 
                       help="Compare with other frameworks")
    parser.add_argument("--work-dir", default="./resnet_tune_logs", 
                       help="Directory for tuning logs")
    parser.add_argument("--output", default="resnet_benchmark_results.json", 
                       help="Output file for results")
    parser.add_argument("--num-runs", type=int, default=100, 
                       help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Input shape for ImageNet
    input_shape = (args.batch_size, 3, 224, 224)
    
    # Create benchmark
    benchmark = ResNetBenchmark(
        model_name=args.model,
        batch_size=args.batch_size,
        input_shape=input_shape,
        target=args.target,
        enable_tuning=args.tune
    )
    
    # Run benchmark
    results = benchmark.run_full_benchmark(args.work_dir)
    
    # Add additional framework comparisons
    if args.compare_frameworks:
        input_data = benchmark.input_data.numpy()
        
        # ONNX Runtime comparison
        onnx_results = benchmark_with_onnxruntime(args.model, input_data, args.num_runs)
        if onnx_results:
            results["onnx_runtime"] = onnx_results
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("RESNET BENCHMARK SUMMARY")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Target: {args.target}")
    print(f"Auto-tuning: {'Enabled' if args.tune else 'Disabled'}")
    print()
    
    if "tvm_relax" in results:
        tvm_lat = results["tvm_relax"]["avg_latency_ms"]
        tvm_thr = results["tvm_relax"]["throughput_samples_per_sec"]
        print(f"TVM Relax:     {tvm_lat:.2f}ms  ({tvm_thr:.1f} samples/sec)")
    
    if "pytorch" in results:
        pt_lat = results["pytorch"]["avg_latency_ms"]
        pt_thr = results["pytorch"]["throughput_samples_per_sec"]
        print(f"PyTorch:       {pt_lat:.2f}ms  ({pt_thr:.1f} samples/sec)")
    
    if "onnx_runtime" in results:
        ort_lat = results["onnx_runtime"]["avg_latency_ms"]
        ort_thr = results["onnx_runtime"]["throughput_samples_per_sec"]
        print(f"ONNX Runtime:  {ort_lat:.2f}ms  ({ort_thr:.1f} samples/sec)")
    
    if "speedup" in results:
        print(f"\nTVM Relax Speedup: {results['speedup']:.2f}x vs PyTorch")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()