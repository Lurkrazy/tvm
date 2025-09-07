# Relay VM MetaSchedule Compilation and Benchmarking

This repository contains tools to compile and benchmark Relay models using MetaSchedule database and Relay Virtual Machine (VM).

## Overview

The main script `compile_and_benchmark_relay_vm.py` does the following:

1. **Loads a MetaSchedule JSON database** containing workloads and tuning records
2. **Loads a Relay model** from ONNX or Python file
3. **Applies best schedules** from the database during compilation
4. **Compiles with Relay VM** (not graph executor)
5. **Benchmarks execution** and reports latency statistics
6. **Saves artifacts** including compiled module, TIR dumps, and PTX/CUDA code

## Requirements

- TVM with CUDA support (for GPU targets)
- Python packages: `numpy`, `onnx` (for ONNX models)
- MetaSchedule database files (JSON format)

## Basic Usage

### CUDA GPU Example (Ada Lovelace - SM_89)

```bash
python compile_and_benchmark_relay_vm.py \
  --model resnet18.onnx \
  --db-workload database_workload.json \
  --db-record database_tuning_record.json \
  --target "cuda -arch=sm_89" \
  --device-id 0 \
  --input data:1x3x224x224:float32 \
  --warmup 10 \
  --number 10 \
  --repeat 10 \
  --output-dir ./results
```

### Multiple Inputs Example

```bash
python compile_and_benchmark_relay_vm.py \
  --model model.onnx \
  --db-workload database_workload.json \
  --db-record database_tuning_record.json \
  --target "cuda -arch=sm_89" \
  --input input0:1x3x224x224:float16 \
  --input input1:1x512:int32 \
  --input attention_mask:1x512:float16 \
  --warmup 5 \
  --number 20 \
  --repeat 5
```

### CPU Example

```bash
python compile_and_benchmark_relay_vm.py \
  --model model.onnx \
  --db-workload database_workload.json \
  --db-record database_tuning_record.json \
  --target "llvm -mcpu=core-avx2" \
  --input data:1x3x224x224:float32 \
  --warmup 5 \
  --number 10 \
  --repeat 10
```

### Python Model Example

```bash
python compile_and_benchmark_relay_vm.py \
  --model my_model.py \
  --db-workload database_workload.json \
  --db-record database_tuning_record.json \
  --target "cuda -arch=sm_89" \
  --input data:1x784:float32 \
  --warmup 10 \
  --number 10 \
  --repeat 10
```

## Command Line Arguments

### Required Arguments

- `--model`: Path to model (ONNX file or Python file that defines a Relay IRModule)
- `--db-workload`: Path to `database_workload.json` 
- `--db-record`: Path to `database_tuning_record.json`
- `--target`: Target string (e.g., `"cuda -arch=sm_89"`, `"llvm"`)
- `--input`: Input specification in format `name:shape:dtype` (can be repeated)

### Optional Arguments

- `--device-id`: Device ID (default: 0)
- `--warmup`: Number of warmup runs (default: 10)
- `--number`: Number of runs per repeat (default: 10) 
- `--repeat`: Number of repeats (default: 10)
- `--output-dir`: Output directory for artifacts (default: ./output)
- `--log-level`: Logging level - DEBUG, INFO, WARNING, ERROR (default: INFO)

## Input Specification Format

The `--input` argument accepts specifications in the format: `name:shape:dtype`

**Examples:**
- `data:1x3x224x224:float32` - NCHW image tensor
- `input_ids:1x512:int32` - Integer sequence  
- `attention_mask:1x512:float16` - Half-precision attention mask
- `features:32x768:float32` - Feature matrix

**Supported dtypes:**
- `float32`, `float16` (fp16), `bfloat16` (bf16)
- `int8`, `int16`, `int32`, `int64`
- `uint8`, `uint16`, `uint32`, `uint64`

## Python Model Format

If using a Python file instead of ONNX, the file should define one of these functions:

```python
def get_model():
    """Return (mod, params) tuple"""
    return mod, params

def get_workload():
    """Return (mod, params) tuple"""  
    return mod, params

def build_model():
    """Return (mod, params) tuple"""
    return mod, params
```

Or define module-level variables:
```python
mod = tvm.IRModule(...)  # Required
params = {...}           # Optional
```

## Output Artifacts

The script saves the following artifacts to the output directory:

1. **`timing_results.json`** - Detailed timing statistics
2. **`vm_executable.so`** - Compiled VM executable  
3. **`lowered_tir.txt`** - TIR (Tensor IR) source code
4. **`cuda_kernel.ptx`** - PTX assembly code (CUDA targets only)
5. **`ptx_analysis.json`** - Analysis of tensor core usage (CUDA targets only)

## Timing Statistics

The script reports the following timing metrics:

- **Mean latency** - Average execution time
- **Standard deviation** - Timing variance
- **Percentiles** - p50 (median), p90, p95, p99
- **Min/Max** - Fastest and slowest runs
- **Number of runs** - Total measurements taken

Example output:
```
==================================================
TIMING RESULTS
==================================================
Mean latency:     1.234 ms
Std deviation:    0.045 ms  
Median (p50):     1.230 ms
90th percentile:  1.280 ms
95th percentile:  1.310 ms
99th percentile:  1.350 ms
Min latency:      1.201 ms
Max latency:      1.387 ms
Number of runs:   100
==================================================
```

## CUDA/PTX Analysis

For CUDA targets, the script automatically analyzes generated PTX code for:

- **Tensor core usage** - Detection of WMMA/MMA instructions
- **Data types** - FP16, BF16, INT8, FP32 tensor core operations
- **Compute capability** - SM architecture version
- **Shared memory usage** - Memory allocation patterns

The analysis helps verify that MetaSchedule successfully applied tensor core optimizations.

## Tensor Core Support

The script supports and detects the following tensor core configurations:

| Architecture | Tensor Cores | Supported Types | Instructions |
|--------------|--------------|-----------------|-------------|
| Volta (SM_70) | V1 | FP16 | WMMA |
| Turing (SM_75) | V2 | FP16, INT8 | WMMA |  
| Ampere (SM_80+) | V3 | FP16, BF16, INT8, FP32 | MMA |
| Ada/Hopper (SM_89+) | V4 | FP16, BF16, INT8, FP32, FP8 | MMA |

## Database Format

The MetaSchedule database consists of two JSON files:

1. **`database_workload.json`** - Contains TIR workload definitions
2. **`database_tuning_record.json`** - Contains tuning traces and performance results

These files are typically generated by MetaSchedule tuning runs.

## Troubleshooting

### Common Issues

1. **"No tuning records found"**
   - Verify database files exist and contain records for your target
   - Check that model workloads match database entries
   - Try with `--log-level DEBUG` for detailed matching info

2. **"ONNX import failed"**
   - Install ONNX: `pip install onnx`
   - Verify ONNX model is valid: `python -c "import onnx; onnx.load('model.onnx')"`

3. **"CUDA device not found"**
   - Check CUDA installation and GPU availability
   - Verify device ID with `nvidia-smi`
   - Try `--device-id 0` explicitly

4. **"Compilation failed"**
   - Check target string syntax: `"cuda -arch=sm_XX"`
   - Verify TVM CUDA support: `tvm.runtime.enabled("cuda")`
   - Try CPU target first: `"llvm"`

### Debug Mode

Run with debug logging for detailed information:

```bash
python compile_and_benchmark_relay_vm.py \
  --log-level DEBUG \
  [other arguments...]
```

This will show:
- Database loading details
- MetaSchedule matching process  
- Compilation passes
- Timing breakdown

## Advanced Usage

### Custom Timing Parameters

For high-precision measurements:

```bash
# More warmup, longer measurement  
python compile_and_benchmark_relay_vm.py \
  --warmup 50 \
  --number 50 \
  --repeat 20 \
  [other arguments...]
```

### Different Architectures

```bash
# Volta V100 (SM_70)
--target "cuda -arch=sm_70"

# Turing RTX 20XX (SM_75)  
--target "cuda -arch=sm_75"

# Ampere A100 (SM_80)
--target "cuda -arch=sm_80" 

# Ada RTX 40XX (SM_89)
--target "cuda -arch=sm_89"

# Hopper H100 (SM_90)
--target "cuda -arch=sm_90"
```

### Multiple GPUs

```bash
# Use GPU 1 instead of GPU 0
--device-id 1
```

## Helper Scripts

### PTX Analysis Tool

Analyze an existing PTX file for tensor core usage:

```bash
python dump_cuda_ptx.py cuda_kernel.ptx
```

This will show:
- Tensor core instruction count
- Data types used  
- Example instructions
- Shared memory patterns

## Integration

This script can be integrated into larger workflows:

```python
# Example integration
import subprocess
import json

result = subprocess.run([
    "python", "compile_and_benchmark_relay_vm.py",
    "--model", "model.onnx", 
    "--db-workload", "db_workload.json",
    "--db-record", "db_record.json", 
    "--target", "cuda -arch=sm_89",
    "--input", "data:1x3x224x224:float32",
    "--output-dir", "results"
], capture_output=True, text=True)

# Load timing results
with open("results/timing_results.json") as f:
    timing = json.load(f)
    
print(f"Mean latency: {timing['mean_ms']} ms")
```

## License

Licensed under the Apache License, Version 2.0. See the LICENSE file for details.