# TVM Tensor Core Lowering Research - Project Summary

## Overview

This research project provides a comprehensive deep dive into how Apache TVM lowers TensorIR (TIR) to NVIDIA Tensor Core instructions, covering the complete pipeline from Python-level tensor intrinsics to final PTX assembly.

## Deliverables

### 1. Comprehensive Research Document
**Location:** `docs/TVM_TIR_to_TensorCore_Lowering_Deep_Research.md`

**Contents:**
- Executive summary with key findings and trigger conditions
- Detailed architecture flow diagram (ASCII art)
- Complete source code file inventory with line-level references
- TIR intrinsic → CodeGen → PTX mapping tables
- Shape/layout/type constraint documentation
- Decision conditions and version differences
- Pitfalls, edge cases, and fallback scenarios
- Validation and testing guidelines

**Key Findings:**
- Two parallel lowering paths: WMMA (C++ API) and MMA (inline PTX)
- sm_89 (Ada) confirmed to support `mma.sync.aligned.m16n8k16.*` instructions
- sm_90 (Hopper) WGMMA support not found in current codebase
- Complete trigger condition documentation (target arch, intrinsic selection, buffer scopes)

### 2. Minimal Reproducible Example (MRE)
**Location:** `examples/tensor_core_mre.py`

**Features:**
- 16x16x16 FP16 GEMM with FP32 accumulation
- Demonstrates both WMMA and MMA lowering paths
- Supports multiple architectures (sm_70, sm_80, sm_89)
- Automatic PTX extraction and analysis
- Tensor Core usage detection and verification
- Correctness testing (if CUDA device available)

**Usage:**
```bash
python3 examples/tensor_core_mre.py
```

**Output:**
- Generated PTX files with `mma.sync.aligned` instructions
- TIR (intermediate representation) files
- Analysis report showing Tensor Core detection

### 3. Documentation Files

**`examples/README.md`**
- Brief overview of examples directory
- Usage instructions for tensor_core_mre.py
- Links to comprehensive documentation

**`SUMMARY.md`** (this file)
- Project overview and deliverable index
- Quick reference guide

## Key Source Code Locations

All file paths are relative to repository root (`/home/runner/work/tvm/tvm/`):

### Python Layer
- **Tensor Intrinsics:** `python/tvm/tir/tensor_intrin/cuda.py`
  - WMMA intrinsics: Lines 805-1371
  - MMA intrinsics: Lines 222-703
  - Helper functions for layout transformations

### C++ Transformation Passes
- **Fragment Inference:** `src/tir/transforms/tensorcore_infer_fragment.cc`
- **Tensor Core Rewrite:** `src/tir/transforms/memhammer_tensorcore_rewrite.cc`
- **Storage Management:** `src/tir/transforms/storage_rewrite.cc`
- **Warp Memory:** `src/tir/transforms/lower_warp_memory.cc`

### CodeGen Layer
- **CUDA CodeGen:** `src/target/source/codegen_cuda.cc`
  - WMMA API emission: Lines 933-991
  - PTX MMA emission: Lines 992-1026
  - PTX ldmatrix: Lines 1065-1095
  - Fragment declarations: Lines 1382-1720
- **PTX Utilities:** `src/target/source/ptx.cc`
  - MMA assembly generation: Lines 542-596
  - Shape parsing/validation: Lines 242-288

### Runtime Support
- **Storage Scopes:** `src/runtime/thread_storage_scope.h`
  - WMMA scope definitions: Lines 56-64, 157-164

## Architecture Flow

```
Python TensorIR (with tensorize)
    ↓
TIR with tensor intrinsic calls
    ↓
TIR transformation passes (InferFragment, StorageRewrite, etc.)
    ↓
CodeGen (pattern matching on builtin calls)
    ↓
C++ WMMA API (nvcuda::wmma::*) OR Inline PTX (mma.sync.aligned.*)
    ↓
NVCC compilation
    ↓
PTX assembly with Tensor Core instructions
    ↓
SASS (native GPU machine code)
```

## Supported Configurations

### GPU Architectures
- **sm_70-sm_75:** Volta, Turing (base Tensor Core support)
- **sm_80-sm_86:** Ampere (adds tf32, bf16, sparsity)
- **sm_89:** Ada Lovelace (same TC ISA as Ampere)
- **sm_90:** Hopper (limited support, no WGMMA)

### Matrix Shapes
- m16n16k16 (fp16, int8)
- m16n8k16 (MMA path variants)
- m16n8k32 (int8)
- m8n8k32 (int4)

### Data Types
- FP16 × FP16 → FP32/FP16
- INT8 × INT8 → INT32
- TF32 × TF32 → FP32 (sm_80+)
- BF16 × BF16 → FP32 (sm_80+)
- FP8 × FP8 → FP32 (sm_89+)

## Trigger Conditions

### Required Elements:
1. **Target:** `cuda -arch=sm_XX` where XX >= 70
2. **Schedule:** Use `tensorize` with registered TensorIntrin
3. **Buffer Scopes:** 
   - WMMA: `wmma.matrix_a`, `wmma.matrix_b`, `wmma.accumulator`
   - MMA: `warp` or `m16n8k8.matrixA/B/C`
4. **Fragment Shapes:** Valid combinations (16×16×16, etc.)
5. **Alignment:** 64-byte alignment (automatic in TVM)

### Path Selection:
- **WMMA Path:** Use `wmma_*` intrinsics → generates C++ `nvcuda::wmma::*` calls
- **MMA Path:** Use `mma_*` intrinsics → generates inline PTX `mma.sync.aligned.*`

## Verification Methods

### 1. Check PTX Assembly
```bash
grep -i "mma.sync\|ldmatrix" /tmp/tvm_tensor_core_mre/*.ptx
```

### 2. Verify TIR Scopes
```python
print(mod.script())  # Look for wmma.* or warp scopes
```

### 3. Profile with Nsight Compute
```bash
ncu --metrics sm__inst_executed_pipe_tensor.avg ./app
```
Tensor pipe utilization > 0% confirms Tensor Core usage.

### 4. Expected PTX Instructions
```ptx
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {...}, [...];
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {...}, {...}, {...}, {...};
```

## Common Pitfalls

1. **Layout Mismatches:** Wrong layout leads to incorrect results
2. **Missing Thread Binding:** Must bind to `threadIdx.x` for warp operations
3. **Invalid Shapes:** Only specific (M,N,K) combinations supported
4. **Type Mismatches:** Not all (A_dtype, B_dtype, C_dtype) combos valid
5. **Alignment Issues:** Fragment buffers must be properly aligned
6. **Fallback to CUDA Cores:** Occurs when constraints not met

## Version Information

- **Commit SHA:** d03d0ba9340c509e983dd7066d3a182ad00e9622
- **Date:** 2025-10-19
- **TVM Branch:** main (latest development)

## Future Work / Known Limitations

### sm_90 (Hopper) WGMMA Support
**Status:** Not implemented in current TVM

**Evidence:**
- No `wgmma` references in Python intrinsic definitions
- No `builtin::wgmma_*` calls in codegen layer
- PTX generation focused on `mma.sync`, not `wgmma.mma_async`

**Required for WGMMA:**
1. New TensorIntrin definitions for async operations
2. Builtin function additions (e.g., `builtin::wgmma_mma_async()`)
3. PTX assembly generation support
4. TMA (Tensor Memory Accelerator) descriptor management
5. Async commit/wait barrier handling

**Workaround:** Use sm_89 target even on H100 to utilize `mma.sync` path

## References

### Documentation
- Main research doc: `docs/TVM_TIR_to_TensorCore_Lowering_Deep_Research.md`
- Examples README: `examples/README.md`

### External Resources
- NVIDIA PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/
- WMMA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma
- TVM Documentation: https://tvm.apache.org/docs/

### Related Files
- Python intrinsics: `python/tvm/tir/tensor_intrin/cuda.py`
- CUDA codegen: `src/target/source/codegen_cuda.cc`
- PTX utilities: `src/target/source/ptx.cc`

## Contact & Contribution

This research was conducted for the `Lurkrazy/tvm` repository.

For questions or contributions:
1. Review the comprehensive documentation in `docs/`
2. Run the MRE script to understand the pipeline
3. Examine generated PTX files for validation
4. Profile with Nsight Compute to verify performance

---

*Document Version: 1.0*  
*Last Updated: 2025-10-19*  
*Commit: d03d0ba9340c509e983dd7066d3a182ad00e9622*
