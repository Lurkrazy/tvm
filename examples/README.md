# TVM Examples

This directory contains example scripts demonstrating various TVM features and capabilities.

## Tensor Core Examples

### tensor_core_mre.py

Minimal reproducible example showing how TVM lowers TensorIR to NVIDIA Tensor Core instructions.

**Features:**
- Demonstrates 16x16x16 FP16 GEMM using WMMA intrinsics
- Shows both WMMA (C++ API) and MMA (inline PTX) paths
- Generates and analyzes PTX assembly
- Tests multiple GPU architectures (sm_70, sm_80, sm_89)

**Usage:**
```bash
python3 examples/tensor_core_mre.py
```

**Requirements:**
- TVM with CUDA support
- CUDA toolkit >= 11.0
- GPU with compute capability >= 7.0 (optional, for correctness testing)

**Output:**
- PTX assembly files in `/tmp/tvm_tensor_core_mre/`
- TIR (Tensor Intermediate Representation) files
- Analysis of Tensor Core usage

**Related Documentation:**
- `docs/TVM_TIR_to_TensorCore_Lowering_Deep_Research.md` - Comprehensive deep dive into the lowering process
