#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Compile and benchmark Relay models using MetaSchedule database and Relay VM.

This script loads a MetaSchedule JSON database, applies the best schedules
to a Relay model, compiles it with Relay VM, and measures latency.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import statistics


def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy is required. Install with: pip install numpy")
    
    try:
        import tvm
        from tvm import relay, runtime
        from tvm.meta_schedule.database import JSONDatabase
        from tvm.meta_schedule.relay_integration import compile_relay
        from tvm.ir import transform
    except ImportError as e:
        raise ImportError(f"TVM is required but not available: {e}")
    
    return True


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


def parse_input_spec(input_spec: str) -> Tuple[str, List[int], str]:
    """Parse input specification in format name:shape:dtype.
    
    Args:
        input_spec: String in format "name:shape:dtype", e.g., "input0:1x3x224x224:float32"
        
    Returns:
        Tuple of (name, shape, dtype)
    """
    parts = input_spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid input spec format: {input_spec}. Expected name:shape:dtype")
    
    name = parts[0]
    shape_str = parts[1]
    dtype = parts[2]
    
    # Parse shape like "1x3x224x224" into [1, 3, 224, 224]
    try:
        shape = [int(dim) for dim in shape_str.split("x")]
    except ValueError as e:
        raise ValueError(f"Invalid shape format in {input_spec}: {e}")
    
    return name, shape, dtype


def load_onnx_model(model_path: str) -> Tuple[Any, Dict[str, Any]]:
    """Load ONNX model and convert to Relay IRModule.
    
    Args:
        model_path: Path to ONNX model file
        
    Returns:
        Tuple of (IRModule, params)
    """
    try:
        import onnx
    except ImportError:
        raise ImportError("ONNX package is required to load ONNX models. Install with: pip install onnx")
    
    # Import TVM here to avoid import at module level
    import tvm
    from tvm import relay
    
    logging.info(f"Loading ONNX model from {model_path}")
    onnx_model = onnx.load(model_path)
    
    # Convert to Relay
    mod, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)
    logging.info(f"Successfully converted ONNX model to Relay IRModule")
    
    return mod, params


def load_python_model(model_path: str) -> Tuple[Any, Dict[str, Any]]:
    """Load model from Python file that returns IRModule.
    
    Args:
        model_path: Path to Python file
        
    Returns:
        Tuple of (IRModule, params)
    """
    # Import TVM here to avoid import at module level
    import tvm
    
    logging.info(f"Loading Python model from {model_path}")
    
    # Add the directory containing the model file to Python path
    model_dir = os.path.dirname(os.path.abspath(model_path))
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    
    # Import the module
    import importlib.util
    spec = importlib.util.spec_from_file_location("model", model_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from {model_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Try to get the model from common function names
    for func_name in ["get_model", "get_workload", "build_model"]:
        if hasattr(module, func_name):
            result = getattr(module, func_name)()
            if isinstance(result, tuple) and len(result) == 2:
                mod, params = result
                return mod, params
            elif isinstance(result, tvm.IRModule):
                return result, {}
    
    # Try to get IRModule directly
    if hasattr(module, "mod"):
        return module.mod, getattr(module, "params", {})
    
    raise ValueError(f"Cannot find model in {model_path}. Expected function get_model(), get_workload(), or build_model()")


def create_random_inputs(input_specs: List[str], device: Any) -> Dict[str, Any]:
    """Create random input tensors based on specifications.
    
    Args:
        input_specs: List of input specifications in format "name:shape:dtype"
        device: Target device
        
    Returns:
        Dictionary mapping input names to NDArrays
    """
    # Import numpy and tvm here
    import numpy as np
    import tvm
    
    inputs = {}
    
    for spec in input_specs:
        name, shape, dtype = parse_input_spec(spec)
        logging.info(f"Creating random input '{name}' with shape {shape} and dtype {dtype}")
        
        # Create random data based on dtype
        if dtype.startswith("float"):
            # Use small values for numerical stability
            data = np.random.uniform(-1.0, 1.0, shape).astype(dtype)
        elif dtype.startswith("int"):
            if "int8" in dtype:
                data = np.random.randint(-128, 127, shape, dtype=dtype)
            elif "int16" in dtype:
                data = np.random.randint(-32768, 32767, shape, dtype=dtype)
            else:
                data = np.random.randint(0, 255, shape, dtype=dtype)
        else:
            # Default to float32
            data = np.random.uniform(-1.0, 1.0, shape).astype("float32")
            logging.warning(f"Unknown dtype {dtype}, using float32")
        
        inputs[name] = tvm.nd.array(data, device=device)
    
    return inputs


def load_database(db_workload_path: str, db_record_path: str) -> Any:
    """Load MetaSchedule JSON database.
    
    Args:
        db_workload_path: Path to database_workload.json
        db_record_path: Path to database_tuning_record.json
        
    Returns:
        JSONDatabase instance
    """
    from tvm.meta_schedule.database import JSONDatabase
    
    logging.info(f"Loading MetaSchedule database:")
    logging.info(f"  Workload: {db_workload_path}")
    logging.info(f"  Records: {db_record_path}")
    
    # Verify files exist
    if not os.path.exists(db_workload_path):
        raise FileNotFoundError(f"Database workload file not found: {db_workload_path}")
    if not os.path.exists(db_record_path):
        raise FileNotFoundError(f"Database record file not found: {db_record_path}")
    
    database = JSONDatabase(
        path_workload=db_workload_path,
        path_tuning_record=db_record_path,
        allow_missing=False
    )
    
    # Log some database statistics
    try:
        with open(db_workload_path, 'r') as f:
            workloads = json.load(f)
        with open(db_record_path, 'r') as f:
            records = json.load(f)
        
        logging.info(f"Database contains {len(workloads)} workloads and {len(records)} tuning records")
    except Exception as e:
        logging.warning(f"Could not read database statistics: {e}")
    
    return database


def compile_model_with_metaschedule(
    mod: Any,
    params: Dict[str, Any],
    target: str,
    database: Any
) -> Any:
    """Compile Relay model with MetaSchedule using VM backend.
    
    Args:
        mod: Relay IRModule
        params: Model parameters
        target: Target string
        database: MetaSchedule database
        
    Returns:
        VM Executable
    """
    import tvm
    from tvm.meta_schedule.relay_integration import compile_relay
    
    logging.info(f"Compiling model with target: {target}")
    
    target_obj = tvm.target.Target(target)
    
    try:
        # Use MetaSchedule relay integration to compile with VM backend
        executable = compile_relay(
            database=database,
            mod=mod,
            target=target_obj,
            params=params,
            backend="vm",
            opt_level=3,
            pass_config={
                "relay.backend.use_meta_schedule": True,
                "relay.backend.tir_converter": "default",
            }
        )
        
        logging.info("Successfully compiled model with MetaSchedule")
        return executable
        
    except Exception as e:
        logging.error(f"Failed to compile model with MetaSchedule: {e}")
        raise


def benchmark_vm(
    executable: Any,
    inputs: Dict[str, Any],
    device: Any,
    warmup: int = 10,
    number: int = 10,
    repeat: int = 10
) -> Dict[str, float]:
    """Benchmark VM execution and return timing statistics.
    
    Args:
        executable: VM executable
        inputs: Input tensors
        device: Target device
        warmup: Number of warmup runs
        number: Number of runs per repeat
        repeat: Number of repeats
        
    Returns:
        Dictionary with timing statistics in milliseconds
    """
    import numpy as np
    import tvm
    
    logging.info(f"Starting benchmark with warmup={warmup}, number={number}, repeat={repeat}")
    
    # Create VM instance
    vm = tvm.runtime.vm.VirtualMachine(executable, device)
    
    # Convert inputs to list in correct order
    input_list = list(inputs.values())
    
    # Warmup runs
    logging.info("Running warmup...")
    for _ in range(warmup):
        vm.run(*input_list)
    
    # Actual timing
    logging.info("Running timed benchmark...")
    
    try:
        # Use VM's time_evaluator
        timer = vm.module.time_evaluator("invoke", device, number=number, repeat=repeat)
        timing_results = timer("main", *input_list)
        
        # Convert to milliseconds
        times_ms = np.array(timing_results.results) * 1000.0
        
        # Calculate statistics
        stats = {
            "mean_ms": float(np.mean(times_ms)),
            "std_ms": float(np.std(times_ms)),
            "p50_ms": float(np.percentile(times_ms, 50)),
            "p90_ms": float(np.percentile(times_ms, 90)),
            "p95_ms": float(np.percentile(times_ms, 95)),
            "p99_ms": float(np.percentile(times_ms, 99)),
            "min_ms": float(np.min(times_ms)),
            "max_ms": float(np.max(times_ms)),
            "num_runs": len(times_ms)
        }
        
        logging.info("Benchmark completed successfully")
        return stats
        
    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        raise


def print_timing_stats(stats: Dict[str, float]) -> None:
    """Print timing statistics in a formatted manner."""
    print("\n" + "="*50)
    print("TIMING RESULTS")
    print("="*50)
    print(f"Mean latency:     {stats['mean_ms']:.3f} ms")
    print(f"Std deviation:    {stats['std_ms']:.3f} ms")
    print(f"Median (p50):     {stats['p50_ms']:.3f} ms")
    print(f"90th percentile:  {stats['p90_ms']:.3f} ms")
    print(f"95th percentile:  {stats['p95_ms']:.3f} ms")
    print(f"99th percentile:  {stats['p99_ms']:.3f} ms")
    print(f"Min latency:      {stats['min_ms']:.3f} ms")
    print(f"Max latency:      {stats['max_ms']:.3f} ms")
    print(f"Number of runs:   {stats['num_runs']}")
    print("="*50 + "\n")


def save_artifacts(
    executable: Any,
    stats: Dict[str, float],
    output_dir: str,
    target: str
) -> None:
    """Save compilation and benchmark artifacts.
    
    Args:
        executable: VM executable
        stats: Timing statistics
        output_dir: Output directory
        target: Target string
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Saving artifacts to {output_dir}")
    
    # Save timing results as JSON
    timing_file = output_path / "timing_results.json"
    with open(timing_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Saved timing results to {timing_file}")
    
    # Save compiled VM executable
    try:
        vm_file = output_path / "vm_executable.so"
        executable.mod.export_library(str(vm_file))
        logging.info(f"Saved VM executable to {vm_file}")
    except Exception as e:
        logging.warning(f"Could not save VM executable: {e}")
    
    # Extract and save TIR code
    try:
        tir_file = output_path / "lowered_tir.txt"
        mod = executable.mod
        
        # Get the IRModule if available
        if hasattr(mod, 'get_source'):
            tir_source = mod.get_source()
            with open(tir_file, 'w') as f:
                f.write(tir_source)
            logging.info(f"Saved TIR source to {tir_file}")
    except Exception as e:
        logging.warning(f"Could not extract TIR source: {e}")
    
    # Extract CUDA/PTX code if available
    if "cuda" in target.lower():
        try:
            ptx_file = output_path / "cuda_ptx.ptx"
            
            # Try to get PTX source
            ptx_source = None
            
            # Method 1: Check imported modules for PTX
            if hasattr(executable.mod, 'imported_modules'):
                for imported_mod in executable.mod.imported_modules:
                    if hasattr(imported_mod, 'get_source'):
                        try:
                            ptx_source = imported_mod.get_source("ptx")
                            if ptx_source:
                                break
                        except:
                            try:
                                ptx_source = imported_mod.get_source()
                                if ptx_source and ".version" in ptx_source:  # PTX signature
                                    break
                            except:
                                continue
            
            if ptx_source:
                with open(ptx_file, 'w') as f:
                    f.write(ptx_source)
                logging.info(f"Saved PTX source to {ptx_file}")
            else:
                logging.warning("Could not extract PTX source")
                
        except Exception as e:
            logging.warning(f"Could not extract PTX source: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compile and benchmark Relay models using MetaSchedule database and Relay VM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python compile_and_benchmark_relay_vm.py \\
    --model model.onnx \\
    --db-workload database_workload.json \\
    --db-record database_tuning_record.json \\
    --target "cuda -arch=sm_89" \\
    --input input0:1x3x224x224:float32 \\
    --warmup 5 --number 10 --repeat 10
        """
    )
    
    # Model and database arguments
    parser.add_argument("--model", required=True, help="Path to model (ONNX file or Python file)")
    parser.add_argument("--db-workload", required=True, help="Path to database_workload.json")
    parser.add_argument("--db-record", required=True, help="Path to database_tuning_record.json")
    
    # Target and device
    parser.add_argument("--target", required=True, help="Target string, e.g., 'cuda -arch=sm_89'")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID (default: 0)")
    
    # Input specifications
    parser.add_argument("--input", action="append", required=True,
                       help="Input specification: name:shape:dtype (can be repeated)")
    
    # Timing parameters
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs (default: 10)")
    parser.add_argument("--number", type=int, default=10, help="Number of runs per repeat (default: 10)")
    parser.add_argument("--repeat", type=int, default=10, help="Number of repeats (default: 10)")
    
    # Output
    parser.add_argument("--output-dir", default="./output", help="Output directory for artifacts (default: ./output)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    
    # Setup logging first
    setup_logging(args.log_level)
    
    # Check dependencies
    try:
        check_dependencies()
        logging.info("All dependencies are available")
    except ImportError as e:
        logging.error(f"Dependency check failed: {e}")
        sys.exit(1)
    
    # Import dependencies after checking
    import numpy as np
    import tvm
    from tvm import relay, runtime
    from tvm.meta_schedule.database import JSONDatabase
    from tvm.meta_schedule.relay_integration import compile_relay
    
    try:
        # Load model
        model_path = args.model
        if model_path.endswith(".onnx"):
            mod, params = load_onnx_model(model_path)
        else:
            mod, params = load_python_model(model_path)
        
        # Load database
        database = load_database(args.db_workload, args.db_record)
        
        # Setup device
        if "cuda" in args.target:
            device = tvm.cuda(args.device_id)
        elif "opencl" in args.target:
            device = tvm.opencl(args.device_id)
        elif "metal" in args.target:
            device = tvm.metal(args.device_id)
        else:
            device = tvm.cpu(args.device_id)
        
        logging.info(f"Using device: {device}")
        
        # Create random inputs
        inputs = create_random_inputs(args.input, device)
        
        # Compile model
        executable = compile_model_with_metaschedule(mod, params, args.target, database)
        
        # Benchmark
        stats = benchmark_vm(executable, inputs, device, args.warmup, args.number, args.repeat)
        
        # Print results
        print_timing_stats(stats)
        
        # Save artifacts
        save_artifacts(executable, stats, args.output_dir, args.target)
        
        logging.info("Script completed successfully")
        
    except Exception as e:
        logging.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()