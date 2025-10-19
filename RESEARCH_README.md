# TVM TIR → Tensor Core (WMMA) Lowering - Deep Research

**Repository:** Lurkrazy/tvm  
**Branch:** copilot/research-tvm-tir-lowering  
**Commit:** 1966fa1  
**Date:** 2025-10-19  

---

## 🎯 Research Objective

This research provides a **complete, source-code-grounded investigation** of how Apache TVM lowers TensorIR (TIR) to NVIDIA Tensor Core instructions, covering the entire pipeline from Python-level tensor intrinsics to final PTX assembly code.

## 📚 Complete Documentation Set

### Core Research Documents

1. **[Main Research Document](docs/TVM_TIR_to_TensorCore_Lowering_Deep_Research.md)** (30KB)
   - Executive summary with key findings
   - Complete end-to-end architecture flow diagram
   - Source code file structure with line-level references
   - TIR intrinsic → CodeGen → PTX mapping tables
   - Shape/layout/type constraints
   - Decision conditions and version differences
   - Pitfalls, edge cases, and fallback scenarios
   - Validation and testing guidelines

2. **[Visual Flow Diagrams](docs/TVM_TensorCore_Lowering_Flow_Diagrams.md)** (30KB)
   - High-level architecture overview (detailed ASCII diagram)
   - Decision tree for WMMA vs MMA path selection
   - Fragment memory hierarchy visualization
   - Supported matrix shapes and types matrix
   - Validation checklist with step-by-step verification

3. **[Project Summary](SUMMARY.md)** (7KB)
   - Quick reference guide
   - Deliverables index
   - Key source code locations
   - Verification methods
   - Contact and contribution information

### Executable Code

4. **[Minimal Reproducible Example](examples/tensor_core_mre.py)** (14KB)
   - 16×16×16 FP16 GEMM demonstration
   - Both WMMA (C++ API) and MMA (inline PTX) paths
   - Multi-architecture support (sm_70, sm_80, sm_89)
   - Automatic PTX extraction and analysis
   - Tensor Core usage detection
   - Correctness testing (if CUDA device available)

5. **[Examples README](examples/README.md)**
   - Usage instructions
   - Requirements
   - Expected output

## 🚀 Quick Start

### Run the Minimal Reproducible Example

```bash
# Clone the repository (if not already done)
git clone https://github.com/Lurkrazy/tvm.git
cd tvm
git checkout copilot/research-tvm-tir-lowering

# Run the MRE (demonstrates TIR → Tensor Core lowering)
python3 examples/tensor_core_mre.py
```

**Expected Output:**
- Generated PTX files in `/tmp/tvm_tensor_core_mre/`
- Analysis showing `mma.sync.aligned` instructions detected
- Verification of Tensor Core usage for each architecture

### Examine Generated PTX

```bash
# View generated PTX files
ls -lh /tmp/tvm_tensor_core_mre/

# Search for Tensor Core instructions
grep -i "mma.sync\|ldmatrix" /tmp/tvm_tensor_core_mre/*.ptx

# View specific architecture PTX
cat /tmp/tvm_tensor_core_mre/wmma_gemm_s89.ptx
```

## 🔍 Key Findings

### ✅ Complete Pipeline Documented

**From TensorIR to PTX in 6 Steps:**

```
1. Python TensorIntrin Registration
   ↓ (python/tvm/tir/tensor_intrin/cuda.py)
   
2. TIR Generation with Intrinsic Calls
   ↓ (via tensorize schedule primitive)
   
3. TIR Transformation Passes
   ↓ (InferFragment, StorageRewrite, LowerWarpMemory, etc.)
   
4. CodeGen Pattern Matching
   ↓ (src/target/source/codegen_cuda.cc)
   
5. C++ WMMA API or Inline PTX Generation
   ↓ (nvcuda::wmma::* or __asm__ __volatile__)
   
6. Final PTX with mma.sync.aligned Instructions
   ↓ (compiled by nvcc)
```

### ✅ Two Parallel Lowering Paths

| Aspect | WMMA Path (Legacy) | MMA Path (Modern) |
|--------|-------------------|-------------------|
| **Intrinsics** | `wmma_load_*`, `wmma_sync_*`, `wmma_fill_*`, `wmma_store_*` | `mma_ldmatrix_*`, `mma_f16f16f32`, `mma_fill_*`, `mma_store_*` |
| **Buffer Scopes** | `wmma.matrix_a`, `wmma.matrix_b`, `wmma.accumulator` | `warp`, `m16n8k8.matrixA/B/C` |
| **TIR Builtins** | `tvm_load_matrix_sync`, `tvm_mma_sync`, etc. | `ptx_ldmatrix`, `ptx_mma`, etc. |
| **CodeGen Output** | `nvcuda::wmma::*` C++ API calls | Inline PTX `__asm__ __volatile__` |
| **PTX Instructions** | `mma.sync.aligned.*` (via WMMA API) | `mma.sync.aligned.*` (direct) |
| **Advantages** | Stable, portable across CUDA versions | More control, direct PTX generation |

### ✅ Architecture Support Confirmed

| Architecture | Support Status | PTX Instructions | Notes |
|-------------|----------------|------------------|-------|
| **sm_70-sm_75** (Volta, Turing) | ✅ Full Support | `mma.sync.aligned.m16n16k16.*` | Base Tensor Core support |
| **sm_80-sm_86** (Ampere) | ✅ Full Support | `mma.sync.aligned.m16n8k16.*` | Adds tf32, bf16, sparsity |
| **sm_89** (Ada, RTX 4090) | ✅ Full Support | `mma.sync.aligned.m16n8k16.*` | Same TC ISA as Ampere |
| **sm_90** (Hopper, H100) | ❌ Limited Support | No WGMMA found | Can use mma.sync fallback |

### ❌ sm_90 (Hopper) WGMMA Status

**Finding:** WGMMA (warp group matrix multiply-accumulate) support **NOT FOUND** in current TVM codebase.

**Evidence:**
- No `wgmma` string in Python intrinsic definitions (`python/tvm/tir/tensor_intrin/cuda.py`)
- No `builtin::wgmma_*` calls in codegen (`src/target/source/codegen_cuda.cc`)
- PTX generation focused on `mma.sync`, not `wgmma.mma_async`

**Workaround:** Use `-arch=sm_89` target even on H100 GPUs to utilize existing `mma.sync` path.

**Future Work Needed:**
1. New TensorIntrin definitions for WGMMA async operations
2. Builtin function additions (e.g., `builtin::wgmma_mma_async()`)
3. PTX assembly generation support in `src/target/source/ptx.cc`
4. TMA (Tensor Memory Accelerator) descriptor management
5. Async barrier and commit/wait handling

## 📊 Source Code Reference Map

All paths relative to repository root:

### Python Layer (Intrinsic Definitions)
```
python/tvm/tir/tensor_intrin/cuda.py
├── Lines 222-283:   MMA ldmatrix intrinsics (mma_ldmatrix_f16_a/b, etc.)
├── Lines 286-544:   MMA sync intrinsics (mma_f16f16f32, mma_i8i8i32, etc.)
├── Lines 547-703:   MMA fill/store intrinsics
├── Lines 805-894:   WMMA load intrinsics (wmma_load_16x16x16_*)
├── Lines 897-951:   WMMA fill intrinsics (wmma_fill_16x16x16_*)
├── Lines 953-1022:  WMMA store intrinsics (wmma_store_16x16x16_*)
└── Lines 1024-1174: WMMA sync intrinsics (wmma_sync_16x16x16_*)
```

### C++ CodeGen Layer
```
src/target/source/codegen_cuda.cc
├── Lines 933-942:   tvm_fill_fragment → nvcuda::wmma::fill_fragment()
├── Lines 943-954:   tvm_load_matrix_sync → nvcuda::wmma::load_matrix_sync()
├── Lines 955-971:   tvm_store_matrix_sync → nvcuda::wmma::store_matrix_sync()
├── Lines 972-981:   tvm_mma_sync → nvcuda::wmma::mma_sync()
├── Lines 992-1026:  ptx_mma → PrintMMAAssembly() → inline PTX
├── Lines 1065-1095: ptx_ldmatrix → PrintLoadMatrixAssembly() → inline PTX
└── Lines 1382-1720: Fragment type declarations (wmma.matrix_a/b/accumulator)

src/target/source/ptx.cc
├── Lines 82-149:    DTypeFromString() - Data type conversions
├── Lines 242-288:   ParseMMAShape() - Shape parsing and validation
└── Lines 542-596:   PrintMMAAssembly() - PTX assembly generation
```

### TIR Transformation Passes
```
src/tir/transforms/tensorcore_infer_fragment.cc
└── InferFragment() pass - Extracts and validates fragment metadata

src/tir/transforms/memhammer_tensorcore_rewrite.cc
└── Lines 96-192: RewriteTensorCoreLoad() - Rewrites data copies to fragments

src/tir/transforms/storage_rewrite.cc
└── Buffer allocation and rewriting for tensor core fragments

src/runtime/thread_storage_scope.h
└── Lines 56-64, 157-164: WMMA scope definitions
```

## 🔧 Trigger Conditions & Usage

### Required Elements for Tensor Core Usage

1. **Target Architecture:**
   ```python
   target = tvm.target.Target("cuda -arch=sm_89")  # sm_70 or higher
   ```

2. **Schedule with Tensorize:**
   ```python
   s = te.create_schedule(C.op)
   s.tensorize(block, tvm.tir.TensorIntrin.get("wmma_sync_16x16x16_f16f16f32"))
   ```

3. **Buffer Scopes:**
   ```python
   # WMMA Path
   A_frag = s.cache_read(A_shared, "wmma.matrix_a", [C_local])
   B_frag = s.cache_read(B_shared, "wmma.matrix_b", [C_local])
   C_local = s.cache_write(C, "wmma.accumulator")
   
   # MMA Path
   A_warp = s.cache_read(A_shared, "warp", [C_warp])
   B_warp = s.cache_read(B_shared, "warp", [C_warp])
   C_warp = s.cache_write(C, "warp")
   ```

4. **Valid Fragment Shapes:**
   - m16n16k16, m16n8k16, m16n8k32, m8n8k32

5. **Supported Data Types:**
   - FP16 × FP16 → FP32/FP16
   - INT8 × INT8 → INT32
   - TF32 × TF32 → FP32 (sm_80+)
   - FP8 × FP8 → FP32 (sm_89+)

## ✅ Validation & Verification

### Method 1: Check PTX Assembly

```bash
# Extract PTX from built module
python3 -c "
import tvm
# ... build module ...
ptx = mod.imported_modules[0].get_source()
print(ptx)
" > output.ptx

# Search for Tensor Core instructions
grep -i "mma.sync\|ldmatrix" output.ptx
```

**Expected PTX Instructions:**
```ptx
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0, %r1, %r2, %r3}, [%rd0];
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {...}, {...}, {...}, {...};
```

### Method 2: Inspect TIR

```python
# Print the TIR to verify buffer scopes and intrinsic calls
print(mod.script())

# Look for:
# - Buffer scopes: wmma.matrix_a, wmma.matrix_b, wmma.accumulator (WMMA path)
# - Buffer scopes: warp, m16n8k8.matrixA/B/C (MMA path)
# - Intrinsic calls: T.tvm_mma_sync, T.ptx_mma, T.ptx_ldmatrix
```

### Method 3: Profile with Nsight Compute

```bash
# Compile your application with TVM
# Then profile to verify Tensor Core utilization

ncu --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active ./your_app

# Tensor pipe utilization should be > 0% if Tensor Cores are used
```

### Method 4: Run the MRE

```bash
# The MRE automatically validates Tensor Core usage
python3 examples/tensor_core_mre.py

# Output will show:
# ✅ TENSOR CORES DETECTED
# Found X mma.sync.aligned instruction(s)
# Found Y ldmatrix instruction(s)
```

## 🚨 Common Pitfalls

1. **Layout Mismatches** 
   - Issue: Wrong A/B layout specification
   - Result: Incorrect computation results (not build errors)
   - Fix: Use correct intrinsic variant (`_trans` suffix for transposed B)

2. **Invalid Fragment Shapes**
   - Issue: Non-standard (M, N, K) dimensions
   - Result: Fallback to regular CUDA cores
   - Fix: Tile loops to valid fragment sizes (16×16×16, etc.)

3. **Missing Thread Binding**
   - Issue: No `threadIdx.x` binding for warp operations
   - Result: Compilation failure or runtime errors
   - Fix: Ensure `s[stage].bind(axis, te.thread_axis("threadIdx.x"))`

4. **Type Mismatches**
   - Issue: Unsupported (A_dtype, B_dtype, C_dtype) combination
   - Result: Build error or fallback
   - Fix: Use validated type combinations (see documentation)

5. **Alignment Issues**
   - Issue: Shared memory not 128-byte aligned for ldmatrix
   - Result: Runtime errors or incorrect results
   - Fix: TVM handles automatically; check `align=64` in match_buffer

6. **Fallback Detection**
   - Issue: Tensor Core path not triggered
   - Result: PTX contains only `fma.rn.f32` instead of `mma.sync`
   - Fix: Verify all trigger conditions are met

## 📖 Additional Resources

### TVM Documentation
- TVM Official Docs: https://tvm.apache.org/docs/
- TensorIR Tutorial: https://tvm.apache.org/docs/tutorial/tensor_ir_intro.html

### NVIDIA Documentation
- PTX ISA Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/
- WMMA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma
- Tensor Cores Performance Guide: https://docs.nvidia.com/deeplearning/performance/

### Related Files in This Repository
- [Main Research Doc](docs/TVM_TIR_to_TensorCore_Lowering_Deep_Research.md)
- [Flow Diagrams](docs/TVM_TensorCore_Lowering_Flow_Diagrams.md)
- [Project Summary](SUMMARY.md)
- [MRE Script](examples/tensor_core_mre.py)

## 🤝 Contributing

This research was conducted for the `Lurkrazy/tvm` repository as a comprehensive deep dive into TVM's Tensor Core lowering pipeline.

For questions or contributions:
1. Review the comprehensive documentation in `docs/`
2. Run the MRE script to understand the pipeline: `python3 examples/tensor_core_mre.py`
3. Examine generated PTX files for validation
4. Profile with Nsight Compute to verify performance

## 📝 Citation

If you use this research, please reference:

```
TVM TIR → Tensor Core (WMMA) Lowering - Deep Research
Repository: Lurkrazy/tvm
Branch: copilot/research-tvm-tir-lowering
Commit: 1966fa1
Date: 2025-10-19
```

---

**Version:** 1.0  
**Last Updated:** 2025-10-19  
**Commit SHA:** 1966fa1  
**Total Documentation:** ~85KB across 5 files  
**Lines of Analysis:** 2000+ source code lines documented  
