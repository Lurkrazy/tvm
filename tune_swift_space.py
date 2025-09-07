#!/usr/bin/env python3
"""
Swift Space Tuning and Benchmarking Driver

A comprehensive driver for tuning TVM MetaSchedule with Swift-style space generator
and benchmarking against default spaces.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import statistics

try:
    import numpy as np
    import tvm
    from tvm import meta_schedule as ms
    from tvm import relay, te, tir
    from tvm.target import Target
    from tvm.contrib import nvcc
    from tvm.tir.schedule import Schedule
    
    # Import our Swift components
    from swift_space_generator import SwiftSpaceGenerator
    from swift_schedule_rules import create_swift_schedule_rules
    from swift_postprocs import create_ordered_swift_postprocessors
    
    TVM_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    TVM_AVAILABLE = False
    print(f"Dependencies not available: {e}")
    
    # Try to import numpy at least
    try:
        import numpy as np
    except ImportError:
        print("Warning: numpy not available either")

logger = logging.getLogger(__name__)


class WorkloadFactory:
    """Factory for creating standard benchmark workloads."""
    
    @staticmethod
    def create_gemm_workload(
        M: int = 1024,
        N: int = 1024, 
        K: int = 1024,
        dtype: str = "float16"
    ) -> Tuple[Any, Any]:
        """Create a GEMM workload for benchmarking."""
        if not TVM_AVAILABLE:
            return None, None
            
        def gemm_func(M, N, K, dtype):
            A = te.placeholder((M, K), name="A", dtype=dtype)
            B = te.placeholder((K, N), name="B", dtype=dtype)
            k = te.reduce_axis((0, K), name="k")
            C = te.compute(
                (M, N),
                lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
                name="C"
            )
            return [A, B, C]
        
        tensors = gemm_func(M, N, K, dtype)
        schedule = te.create_schedule(tensors[-1].op)
        
        # Create IRModule
        func = tvm.build(schedule, tensors, target="llvm", name="gemm")
        mod = tvm.IRModule.from_expr(func.get_source())
        
        return mod, func
    
    @staticmethod
    def create_conv2d_workload(
        N: int = 1,
        H: int = 224,
        W: int = 224,
        C: int = 64,
        K: int = 64,
        R: int = 3,
        S: int = 3,
        stride: int = 1,
        padding: int = 1,
        dtype: str = "float16"
    ) -> Tuple[Any, Any]:
        """Create a Conv2D workload for benchmarking."""
        if not TVM_AVAILABLE:
            return None, None
            
        def conv2d_func(N, H, W, C, K, R, S, stride, padding, dtype):
            data = te.placeholder((N, C, H, W), name="data", dtype=dtype)
            kernel = te.placeholder((K, C, R, S), name="kernel", dtype=dtype)
            
            # Calculate output dimensions
            OH = (H + 2 * padding - R) // stride + 1
            OW = (W + 2 * padding - S) // stride + 1
            
            # Pad input
            pad_h = padding
            pad_w = padding
            padded = te.compute(
                (N, C, H + 2 * pad_h, W + 2 * pad_w),
                lambda n, c, h, w: te.if_then_else(
                    te.all(h >= pad_h, h < H + pad_h, w >= pad_w, w < W + pad_w),
                    data[n, c, h - pad_h, w - pad_w],
                    te.const(0, dtype)
                ),
                name="padded"
            )
            
            # Convolution
            rc = te.reduce_axis((0, C), name="rc")
            rh = te.reduce_axis((0, R), name="rh")
            rw = te.reduce_axis((0, S), name="rw")
            
            conv = te.compute(
                (N, K, OH, OW),
                lambda n, k, oh, ow: te.sum(
                    padded[n, rc, oh * stride + rh, ow * stride + rw] * kernel[k, rc, rh, rw],
                    axis=[rc, rh, rw]
                ),
                name="conv"
            )
            
            return [data, kernel, conv]
        
        tensors = conv2d_func(N, H, W, C, K, R, S, stride, padding, dtype)
        schedule = te.create_schedule(tensors[-1].op)
        
        # Create IRModule
        func = tvm.build(schedule, tensors, target="llvm", name="conv2d")
        mod = tvm.IRModule.from_expr(func.get_source())
        
        return mod, func


class BenchmarkRunner:
    """Runner for benchmarking tuned kernels."""
    
    def __init__(self, target: str = "cuda", dev_id: int = 0):
        self.target_str = target
        self.target = Target(target)
        self.dev_id = dev_id
        
        if TVM_AVAILABLE:
            self.device = tvm.device(target.split()[0], dev_id)
        else:
            self.device = None
    
    def benchmark_module(
        self,
        mod: Any,
        inputs: List[Any],
        outputs: List[Any],
        number: int = 100,
        repeat: int = 10
    ) -> Dict[str, float]:
        """Benchmark a compiled module."""
        if not TVM_AVAILABLE or self.device is None:
            return {}
        
        try:
            # Create timer
            timer = mod.time_evaluator(
                mod.entry_name, self.device, number=number, repeat=repeat
            )
            
            # Run benchmark
            timing_results = timer(*inputs, *outputs)
            times = [float(t) * 1000 for t in timing_results.results]  # Convert to ms
            
            return {
                "mean_ms": statistics.mean(times),
                "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
                "min_ms": min(times),
                "max_ms": max(times),
                "p50_ms": statistics.median(times),
                "p90_ms": np.percentile(times, 90) if len(times) > 1 else times[0],
                "times_ms": times
            }
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {}


class SwiftTuner:
    """Main tuning class for Swift-style space generator."""
    
    def __init__(
        self,
        target: str = "cuda",
        work_dir: str = "./swift_tuning_logs",
        enable_tensorcore: bool = True
    ):
        self.target_str = target
        self.target = Target(target) if TVM_AVAILABLE else None
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        self.enable_tensorcore = enable_tensorcore
        
        # Setup logging
        log_file = self.work_dir / "swift_tuning.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Initialize Swift space generator
        if TVM_AVAILABLE:
            self.swift_factory = SwiftSpaceGenerator(target_str=target)
            self.swift_space = self.swift_factory.create_space_generator()
        else:
            self.swift_factory = None
            self.swift_space = None
    
    def create_default_space(self) -> Optional[Any]:
        """Create TVM's default space generator for comparison."""
        if not TVM_AVAILABLE:
            return None
        
        try:
            # Create default CUDA space
            if "cuda" in self.target_str.lower():
                if self.enable_tensorcore:
                    space = ms.SpaceGenerator.create(
                        kind="post-order-apply",
                        sch_rules="cuda-tensorcore",
                        postprocs="cuda-tensorcore", 
                        mutator_probs="cuda-tensorcore"
                    )
                else:
                    space = ms.SpaceGenerator.create(
                        kind="post-order-apply",
                        sch_rules="cuda",
                        postprocs="cuda",
                        mutator_probs="cuda"
                    )
            else:
                space = ms.SpaceGenerator.create(
                    kind="post-order-apply",
                    sch_rules="llvm",
                    postprocs="llvm",
                    mutator_probs="llvm"
                )
            
            logger.info("Created default space generator")
            return space
            
        except Exception as e:
            logger.error(f"Failed to create default space: {e}")
            return None
    
    def tune_workload(
        self,
        mod: Any,
        trials: int = 1000,
        timeout: int = 10,
        space_generator: Optional[Any] = None,
        space_name: str = "unknown"
    ) -> Optional[Any]:
        """Tune a workload with given space generator."""
        if not TVM_AVAILABLE or space_generator is None:
            return None
        
        try:
            # Setup database
            db_path = self.work_dir / f"{space_name}_database.json"
            database = ms.database.JSONDatabase(work_dir=str(db_path.parent), path_workload=str(db_path))
            
            # Setup tuning context
            tune_context = ms.TuneContext(
                mod=mod,
                target=self.target,
                space_generator=space_generator,
                search_strategy=ms.search_strategy.EvolutionarySearch(),
                cost_model=ms.cost_model.XGBModel(),
                measure_callbacks=[],
                task_name=f"{space_name}_tuning",
                rand_state=42,
            )
            
            # Run tuning
            logger.info(f"Starting tuning with {space_name} space for {trials} trials")
            start_time = time.time()
            
            # Use appropriate tuning API
            if hasattr(ms, 'tune_tir'):
                database = ms.tune_tir(
                    mod=mod,
                    target=self.target,
                    work_dir=str(self.work_dir),
                    max_trials_global=trials,
                    space=space_generator,
                )
            else:
                # Fallback for older TVM versions
                logger.warning("tune_tir not available, skipping tuning")
                return None
            
            tuning_time = time.time() - start_time
            logger.info(f"Tuning completed in {tuning_time:.2f} seconds")
            
            return database
            
        except Exception as e:
            logger.error(f"Tuning failed for {space_name}: {e}")
            return None
    
    def extract_best_schedule(
        self,
        database: Any,
        mod: Any
    ) -> Optional[Tuple[Any, Any]]:
        """Extract the best schedule from tuning database."""
        if not TVM_AVAILABLE or database is None:
            return None, None
        
        try:
            # Get the best record
            best_record = database.query_tuning_record(mod, self.target)
            if not best_record:
                logger.warning("No tuning records found")
                return None, None
            
            # Apply the best trace to get schedule
            sch = tir.Schedule(mod)
            best_record.trace.apply_to_schedule(sch, remove_postproc=False)
            
            # Build the optimized module
            with tvm.transform.PassContext():
                opt_mod = tvm.build(sch.mod, target=self.target)
            
            logger.info("Successfully extracted best schedule")
            return sch, opt_mod
            
        except Exception as e:
            logger.error(f"Failed to extract best schedule: {e}")
            return None, None
    
    def run_comparison(
        self,
        workload_name: str,
        mod: Any,
        inputs: List[Any],
        trials: int = 1000,
        timeout: int = 10
    ) -> Dict[str, Any]:
        """Run A/B comparison between Swift and default spaces."""
        if not TVM_AVAILABLE:
            return {}
        
        results = {
            "workload": workload_name,
            "trials": trials,
            "target": self.target_str,
            "swift": {},
            "default": {},
            "comparison": {}
        }
        
        # Convert inputs to device tensors
        device_inputs = []
        device_outputs = []
        
        if self.swift_space:
            try:
                device = tvm.device(self.target.kind.name, 0)
                for inp in inputs:
                    device_inputs.append(tvm.nd.array(inp, device))
                
                # Create output tensors (assuming single output for now)
                # This is workload-specific and would need refinement
                if workload_name.startswith("gemm"):
                    M, K = inputs[0].shape
                    K2, N = inputs[1].shape
                    output_shape = (M, N)
                    output_dtype = inputs[0].dtype
                elif workload_name.startswith("conv2d"):
                    # Simplified - actual conv2d output shape calculation needed
                    output_shape = (1, 64, 224, 224)  # Example
                    output_dtype = inputs[0].dtype
                else:
                    # Generic fallback
                    output_shape = inputs[0].shape
                    output_dtype = inputs[0].dtype
                
                device_outputs.append(
                    tvm.nd.empty(output_shape, output_dtype, device)
                )
                
            except Exception as e:
                logger.error(f"Failed to prepare device tensors: {e}")
                return results
        
        # Tune with Swift space
        if self.swift_space:
            logger.info("Tuning with Swift space...")
            swift_db = self.tune_workload(
                mod, trials, timeout, self.swift_space, "swift"
            )
            
            if swift_db:
                swift_sch, swift_mod = self.extract_best_schedule(swift_db, mod)
                if swift_mod:
                    runner = BenchmarkRunner(self.target_str)
                    swift_perf = runner.benchmark_module(
                        swift_mod, device_inputs, device_outputs
                    )
                    results["swift"] = {
                        "performance": swift_perf,
                        "tuning_successful": True
                    }
                    
                    # Save artifacts
                    self.save_artifacts(swift_sch, swift_mod, "swift", workload_name)
        
        # Tune with default space
        default_space = self.create_default_space()
        if default_space:
            logger.info("Tuning with default space...")
            default_db = self.tune_workload(
                mod, trials, timeout, default_space, "default"
            )
            
            if default_db:
                default_sch, default_mod = self.extract_best_schedule(default_db, mod)
                if default_mod:
                    runner = BenchmarkRunner(self.target_str)
                    default_perf = runner.benchmark_module(
                        default_mod, device_inputs, device_outputs
                    )
                    results["default"] = {
                        "performance": default_perf,
                        "tuning_successful": True
                    }
                    
                    # Save artifacts
                    self.save_artifacts(default_sch, default_mod, "default", workload_name)
        
        # Compute comparison metrics
        if results["swift"].get("performance") and results["default"].get("performance"):
            swift_time = results["swift"]["performance"]["mean_ms"]
            default_time = results["default"]["performance"]["mean_ms"]
            
            speedup = default_time / swift_time if swift_time > 0 else 0
            
            results["comparison"] = {
                "speedup": speedup,
                "swift_faster": speedup > 1.0,
                "performance_gain_percent": (speedup - 1) * 100
            }
            
            logger.info(f"Swift vs Default speedup: {speedup:.2f}x")
        
        # Save results
        results_file = self.work_dir / f"{workload_name}_comparison.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def save_artifacts(
        self,
        schedule: Optional[Any],
        module: Optional[Any],
        space_name: str,
        workload_name: str
    ):
        """Save tuning artifacts for analysis."""
        if not TVM_AVAILABLE:
            return
        
        artifact_dir = self.work_dir / "artifacts" / f"{workload_name}_{space_name}"
        artifact_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            # Save lowered TIR
            if schedule:
                tir_file = artifact_dir / "lowered.tir"
                with open(tir_file, 'w') as f:
                    f.write(str(schedule.mod))
                logger.info(f"Saved TIR to {tir_file}")
            
            # Save PTX if available (CUDA target)
            if module and "cuda" in self.target_str:
                try:
                    ptx_source = module.imported_modules[0].get_source("ptx")
                    ptx_file = artifact_dir / "kernel.ptx"
                    with open(ptx_file, 'w') as f:
                        f.write(ptx_source)
                    logger.info(f"Saved PTX to {ptx_file}")
                except Exception as e:
                    logger.warning(f"Failed to save PTX: {e}")
            
        except Exception as e:
            logger.error(f"Failed to save artifacts: {e}")


def create_sample_workloads() -> List[Tuple[str, Any, List[Any]]]:
    """Create sample workloads for testing."""
    if not TVM_AVAILABLE:
        return []
    
    workloads = []
    
    # GEMM FP16 1024x1024
    try:
        mod, _ = WorkloadFactory.create_gemm_workload(1024, 1024, 1024, "float16")
        if mod:
            inputs = [
                np.random.randn(1024, 1024).astype(np.float16),
                np.random.randn(1024, 1024).astype(np.float16)
            ]
            workloads.append(("gemm_fp16_1024", mod, inputs))
    except Exception as e:
        logger.warning(f"Failed to create GEMM workload: {e}")
    
    # Conv2D example (simplified)
    try:
        mod, _ = WorkloadFactory.create_conv2d_workload(1, 224, 224, 64, 64, 3, 3, dtype="float16")
        if mod:
            inputs = [
                np.random.randn(1, 64, 224, 224).astype(np.float16),
                np.random.randn(64, 64, 3, 3).astype(np.float16)
            ]
            workloads.append(("conv2d_fp16_224", mod, inputs))
    except Exception as e:
        logger.warning(f"Failed to create Conv2D workload: {e}")
    
    return workloads


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Swift Space Tuning and Benchmarking")
    parser.add_argument("--target", default="cuda -arch=sm_80", help="Target architecture")
    parser.add_argument("--workload", default="auto", help="Workload to tune (auto for samples)")
    parser.add_argument("--trials", type=int, default=1000, help="Number of tuning trials")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout per trial (seconds)")
    parser.add_argument("--work-dir", default="./swift_tuning_logs", help="Working directory")
    parser.add_argument("--no-tensorcore", action="store_true", help="Disable TensorCore")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not TVM_AVAILABLE:
        print("Error: TVM and dependencies not available")
        return 1
    
    # Create tuner
    tuner = SwiftTuner(
        target=args.target,
        work_dir=args.work_dir,
        enable_tensorcore=not args.no_tensorcore
    )
    
    # Run workloads
    if args.workload == "auto":
        workloads = create_sample_workloads()
        if not workloads:
            print("No sample workloads could be created")
            return 1
    else:
        print(f"Custom workload '{args.workload}' not implemented")
        return 1
    
    # Run comparisons
    all_results = []
    for workload_name, mod, inputs in workloads:
        print(f"\n=== Running {workload_name} ===")
        
        result = tuner.run_comparison(
            workload_name, mod, inputs, args.trials, args.timeout
        )
        all_results.append(result)
        
        # Print summary
        if result.get("comparison"):
            comp = result["comparison"]
            print(f"Speedup: {comp['speedup']:.2f}x")
            print(f"Performance gain: {comp['performance_gain_percent']:.1f}%")
    
    # Save consolidated results
    summary_file = Path(args.work_dir) / "swift_benchmark_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {summary_file}")
    print("Swift space tuning completed!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())