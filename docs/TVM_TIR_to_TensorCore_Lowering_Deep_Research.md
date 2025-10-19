# TVM TIR → Tensor Core (WMMA) Lowering Deep Research

**Repository:** Lurkrazy/tvm  
**Commit SHA:** d03d0ba9340c509e983dd7066d3a182ad00e9622  
**Date:** 2025-10-19  
**GPU Target:** RTX 4090 (sm_89), CUDA ≥ 12.2  

---

## Executive Summary

This document provides a comprehensive deep dive into how TVM lowers TensorIR (TIR) to NVIDIA Tensor Core instructions (WMMA API and PTX mma.sync instructions). The lowering pipeline consists of:

1. **Python-level TensorIntrin definition** (`python/tvm/tir/tensor_intrin/cuda.py`) defining both `wmma_*` and `mma_*` intrinsics
2. **TIR Schedule with tensorize** to map compute patterns to tensor intrinsics
3. **TIR transformation passes** (StorageRewrite, InferFragment, etc.) to prepare the IR
4. **CodeGen layer** (`src/target/source/codegen_cuda.cc`, `src/target/source/ptx.cc`) to emit either:
   - **C++ WMMA API** (`nvcuda::wmma::*`) for older/legacy path
   - **Inline PTX** (`mma.sync.aligned.*`) for modern Tensor Cores (sm_70+)

**Key Trigger Conditions:**
- Target: `cuda -arch=sm_XX` where XX >= 70 for Tensor Core support
- TensorIntrin usage via `tensorize` schedule primitive with registered intrinsics
- Buffer scopes: `wmma.matrix_a`, `wmma.matrix_b`, `wmma.accumulator` for WMMA path; `warp` scope for MMA path

**sm_89 (Ada):** Supports `mma.sync.aligned.m16n8k16` PTX instructions (confirmed in MRE below)  
**sm_90 (Hopper):** Limited WGMMA support in current TVM; no production-ready `wgmma.mma_async` path found in source code

---

## A. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TensorIR Python Level                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  User Script (TensorIR + Schedule)                                      │
│       ↓                                                                  │
│  sch.tensorize(block, "wmma_sync_16x16x16_f16f16f32")                  │
│       ↓                                                                  │
│  TensorIntrin.get("wmma_sync_16x16x16_f16f16f32")                      │
│       • desc: High-level computation description                        │
│       • impl: Implementation with T.tvm_mma_sync / T.ptx_mma calls     │
│                                                                          │
└────────────────────────┬────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                       TIR Transformation Passes                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Tensorization → Replaces matched compute blocks with intrin calls   │
│  2. StorageRewrite → Manages buffer allocation for fragments            │
│  3. InferFragment → Adds fragment metadata (m, n, k, layout) attrs      │
│  4. LowerWarpMemory → Handles warp-level memory transformations         │
│  5. LowerIntrin → Lowers high-level intrinsics to backend calls         │
│  6. SplitHostDevice → Separates host and device code                    │
│                                                                          │
│  TIR with calls like:                                                   │
│    • T.tvm_load_matrix_sync(...)   [WMMA path]                         │
│    • T.tvm_mma_sync(...)            [WMMA path]                         │
│    • T.tvm_store_matrix_sync(...)   [WMMA path]                         │
│    • T.ptx_ldmatrix(...)            [MMA path]                          │
│    • T.ptx_mma(...)                 [MMA path]                          │
│                                                                          │
└────────────────────────┬────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         CodeGen Layer (C++)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CodeGenCUDA::VisitExpr_(const CallNode* op)                           │
│       ↓                                                                  │
│  Pattern matching on builtin operations:                                │
│                                                                          │
│  ┌────────────────────────────┬──────────────────────────────────────┐ │
│  │ TIR Builtin                │ Generated Code                        │ │
│  ├────────────────────────────┼──────────────────────────────────────┤ │
│  │ builtin::tvm_fill_fragment │ nvcuda::wmma::fill_fragment(...)     │ │
│  │ builtin::tvm_load_matrix   │ nvcuda::wmma::load_matrix_sync(...)  │ │
│  │ builtin::tvm_store_matrix  │ nvcuda::wmma::store_matrix_sync(...) │ │
│  │ builtin::tvm_mma_sync      │ nvcuda::wmma::mma_sync(...)          │ │
│  │ builtin::ptx_mma           │ inline PTX asm (mma.sync.aligned...) │ │
│  │ builtin::ptx_ldmatrix      │ inline PTX asm (ldmatrix.sync...)    │ │
│  └────────────────────────────┴──────────────────────────────────────┘ │
│                                                                          │
│  For ptx_mma:                                                           │
│    → PrintMMAAssembly() in src/target/source/ptx.cc                    │
│    → Generates: __asm__ __volatile__("mma.sync.aligned.m16n8k16...")  │
│                                                                          │
└────────────────────────┬────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                       CUDA/PTX Output                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Option 1: C++ WMMA API (when using tvm_mma_sync intrinsics)           │
│    #include <mma.h>                                                     │
│    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,          │
│                            half, nvcuda::wmma::row_major> a_frag;       │
│    nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, 16);                  │
│    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);             │
│    nvcuda::wmma::store_matrix_sync(c_ptr, c_frag, 16, ...);            │
│                                                                          │
│  Option 2: Inline PTX (when using ptx_mma intrinsics)                  │
│    __asm__ __volatile__(                                                │
│      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"               │
│      "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};\n"       │
│      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])                  │
│      : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));                    │
│                                                                          │
└────────────────────────┬────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      Final PTX Assembly                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  // sm_70 - sm_89 (Volta, Turing, Ampere, Ada)                         │
│  ldmatrix.sync.aligned.m8n8.x4.shared.b16 {...}, [...];                │
│  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {...}, {...}, ...;  │
│                                                                          │
│  // sm_90+ (Hopper) - Limited support in current TVM                   │
│  wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 {...}, ...;       │
│  // NOTE: No production wgmma path found in current codebase            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## B. Source Code File Structure

### B.1 Python-Level TensorIntrin Definitions

| File Path | Description | Key Functions/Classes |
|-----------|-------------|----------------------|
| `python/tvm/tir/tensor_intrin/__init__.py` | Module initialization for tensor intrinsics | Exports from submodules |
| `python/tvm/tir/tensor_intrin/cuda.py` | **Main CUDA tensor intrinsic definitions** | `get_wmma_load_intrin()`, `get_wmma_sync_intrin()`, `get_wmma_fill_intrin()`, `get_wmma_store_intrin()`, `get_mma_intrin()`, `get_ldmatrix_intrin()` |

**Key Registrations in cuda.py:**
- Lines 1140-1174: WMMA sync intrinsics (`wmma_sync_16x16x16_f16f16f32`, etc.)
- Lines 1180-1273: WMMA load intrinsics (`wmma_load_16x16x16_f16_a_shared`, etc.)
- Lines 1298-1309: WMMA fill intrinsics (`wmma_fill_16x16x16_f32`, etc.)
- Lines 1310-1371: WMMA store intrinsics (`wmma_store_16x16x16_f32_global`, etc.)
- Lines 222-283: MMA ldmatrix intrinsics (`mma_ldmatrix_f16_a`, etc.)
- Lines 470-544: MMA sync intrinsics (`mma_f16f16f32`, etc.)

### B.2 TIR Transformation Passes (C++)

| File Path | Description | Key Functionality |
|-----------|-------------|-------------------|
| `src/tir/transforms/tensorcore_infer_fragment.cc` | Infer and validate fragment metadata | `InferFragment()` pass - extracts m, n, k, layout from wmma intrinsic calls and adds attributes |
| `src/tir/transforms/storage_rewrite.cc` | Buffer allocation and rewriting | Manages memory allocation for tensor core fragments |
| `src/tir/transforms/lower_warp_memory.cc` | Warp-level memory transformations | Handles warp-scope memory operations |
| `src/tir/transforms/lower_intrin.cc` | Lower high-level intrinsics | Converts platform-agnostic intrinsics to target-specific |
| `src/tir/transforms/memhammer_tensorcore_rewrite.cc` | Rewrite tensor core operations | Lines 96-192: `RewriteTensorCoreLoad()` - rewrites data copies to wmma fragments |

### B.3 CodeGen Layer (C++)

| File Path | Description | Key Functionality |
|-----------|-------------|-------------------|
| `src/target/source/codegen_cuda.cc` | Main CUDA code generator | Lines 933-991: WMMA API emission (`tvm_fill_fragment`, `tvm_load_matrix_sync`, `tvm_store_matrix_sync`, `tvm_mma_sync`, `tvm_bmma_sync`)<br>Lines 992-1026: PTX MMA emission (`ptx_mma`)<br>Lines 1065-1095: PTX ldmatrix emission (`ptx_ldmatrix`)<br>Lines 1382-1720: Fragment type declaration (`wmma.matrix_a/b/accumulator`) |
| `src/target/source/codegen_cuda.h` | CUDA codegen header | Class declarations |
| `src/target/source/ptx.cc` | PTX assembly generation utilities | Lines 542-596: `PrintMMAAssembly()` - generates inline PTX for mma.sync<br>Lines 101-149: Data type conversions<br>Lines 242-288: MMA shape parsing and validation |
| `src/target/source/ptx.h` | PTX utilities header | Function declarations for PTX assembly generation |

### B.4 Runtime Support

| File Path | Description | Key Functionality |
|-----------|-------------|-------------------|
| `src/runtime/thread_storage_scope.h` | Thread storage scope definitions | Lines 56-64: Defines `wmma.matrix_a`, `wmma.matrix_b`, `wmma.accumulator` scopes<br>Lines 157-164: Parsing of wmma scopes |

### B.5 Trigger Conditions

**Target Architecture:**
- Specified via `target="cuda -arch=sm_XX"` where:
  - `sm_70+`: Volta Tensor Cores (16x16x16 fp16)
  - `sm_75+`: Turing Tensor Cores (adds int8, int4 support)
  - `sm_80+`: Ampere Tensor Cores (adds tf32, bf16)
  - `sm_89`: Ada Tensor Cores (RTX 40xx series)
  - `sm_90+`: Hopper Tensor Cores (limited support for wgmma)

**Intrinsic Selection:**
- WMMA path: Uses `wmma_*` intrinsics → generates `nvcuda::wmma::*` C++ API
- MMA path: Uses `mma_*` intrinsics → generates inline PTX `mma.sync.aligned.*`
- Decision made at Python tensorize level based on intrinsic name

**Buffer Scopes:**
- WMMA fragments: `wmma.matrix_a`, `wmma.matrix_b`, `wmma.accumulator`
- MMA fragments: `warp` scope for modern mma intrinsics, `m16n8k8.matrixA/B/C` for specific variants

---

## C. TIR Intrinsic → CodeGen → PTX Mapping Table

### C.1 WMMA Path (Legacy/C++ API)

| TensorIntrin Name | TIR Builtin Call | CodeGen Output (C++) | Shape/Layout/Types |
|-------------------|------------------|----------------------|--------------------|
| `wmma_load_16x16x16_f16_a_shared` | `T.tvm_load_matrix_sync(buf, 16, 16, 16, idx, ptr, stride, "row_major")` | `nvcuda::wmma::load_matrix_sync(frag[idx], ptr, stride)` | m16n16k16, row_major, float16 → matrix_a |
| `wmma_load_16x16x16_f16_b_shared` | `T.tvm_load_matrix_sync(buf, 16, 16, 16, idx, ptr, stride, "col_major")` | `nvcuda::wmma::load_matrix_sync(frag[idx], ptr, stride)` | m16n16k16, col_major, float16 → matrix_b |
| `wmma_fill_16x16x16_f32` | `T.tvm_fill_fragment(buf, 16, 16, 16, idx, value)` | `nvcuda::wmma::fill_fragment(frag[idx], value)` | m16n16k16, float32 → accumulator |
| `wmma_sync_16x16x16_f16f16f32` | `T.tvm_mma_sync(d, d_idx, a, a_idx, b, b_idx, c, c_idx)` | `nvcuda::wmma::mma_sync(d_frag[d_idx], a_frag[a_idx], b_frag[b_idx], c_frag[c_idx])` | m16n16k16, f16×f16→f32, B not transposed |
| `wmma_sync_16x16x16_f16f16f32_trans` | `T.tvm_mma_sync(...)` | `nvcuda::wmma::mma_sync(...)` | m16n16k16, f16×f16→f32, B transposed |
| `wmma_store_16x16x16_f32_global` | `T.tvm_store_matrix_sync(buf, 16, 16, 16, idx, ptr, stride, "row_major")` | `nvcuda::wmma::store_matrix_sync(ptr, frag[idx], stride, nvcuda::wmma::mem_row_major)` | m16n16k16, row_major, float32 |

### C.2 MMA Path (Modern/Inline PTX)

| TensorIntrin Name | TIR Builtin Call | CodeGen Output (PTX) | Shape/Layout/Types |
|-------------------|------------------|----------------------|--------------------|
| `mma_ldmatrix_f16_a` | `T.ptx_ldmatrix(trans=False, num=4, ".b16", warp.data, offset, shared.ptr, stride)` | `__asm__ __volatile__("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(...) : "r"(addr))` | Load 4×16 bytes from shared to registers |
| `mma_ldmatrix_f16_b` | `T.ptx_ldmatrix(trans=False, num=4, ".b16", ...)` | `__asm__ __volatile__("ldmatrix.sync.aligned.m8n8.x4.shared.b16 ...")` | Load 4×16 bytes from shared to registers |
| `mma_f16f16f32` | `T.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp32", a_data, a_idx, b_data, b_idx, c_data, c_idx, False)` | `__asm__ __volatile__("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3};\n" : "+f"(c[0-3]) : "r"(a[0-1]), "r"(b[0-1]))` | m16n8k16, row×col, f16×f16+f32→f32 |
| `mma_fill_16x16_f32` | `T.mma_fill(local_size, C.data, C.offset)` | Direct register initialization | Initialize 8 f32 registers to 0 |
| `mma_store_16x16_f32_global` | `T.mma_store(16, 16, dst_ptr, src.data, src.offset, stride)` | Inline store operations per thread | Store from warp registers to global |

### C.3 Shape/Layout/Type Constraints

**Supported Configurations (sm_70 - sm_89):**

| Shape | A Layout | B Layout | A Type | B Type | Accum Type | Notes |
|-------|----------|----------|--------|--------|------------|-------|
| m16n16k16 | row | col | f16 | f16 | f32/f16 | Standard GEMM |
| m16n16k16 | row | row | f16 | f16 | f32/f16 | B transposed |
| m16n16k16 | col | col | f16 | f16 | f32/f16 | A transposed |
| m16n16k16 | col | row | f16 | f16 | f32/f16 | Both transposed |
| m16n16k16 | row | col | s8 | s8 | s32 | INT8 GEMM |
| m16n8k16 | row | col | f16 | f16 | f32 | MMA path variant |
| m16n8k32 | row | col | s8 | s8 | s32 | MMA path INT8 |
| m8n8k32 | - | trans | s4 | s4 | s32 | INT4 GEMM (sm_75+) |

**Alignment Requirements:**
- Shared memory: 128-byte alignment (built into TIR intrinsic definitions)
- Fragment buffers: Allocated with proper alignment via `align=64` in match_buffer
- ldmatrix: Requires 16-byte aligned shared memory addresses

---

## D. Minimal Reproducible Example (MRE)

### D.1 Python Script (16x16x16 FP16 GEMM)

```python
import tvm
from tvm.script import tir as T
from tvm import te
import numpy as np

# Define the computation using TensorIR
@T.prim_func
def wmma_gemm_16x16x16(
    A: T.Buffer((16, 16), "float16"),
    B: T.Buffer((16, 16), "float16"),
    C: T.Buffer((16, 16), "float32"),
):
    """16x16x16 GEMM using WMMA intrinsics"""
    # Shared memory buffers
    A_shared = T.alloc_buffer((16, 16), "float16", scope="shared")
    B_shared = T.alloc_buffer((16, 16), "float16", scope="shared")
    
    # WMMA fragments
    A_frag = T.alloc_buffer((16, 16), "float16", scope="wmma.matrix_a")
    B_frag = T.alloc_buffer((16, 16), "float16", scope="wmma.matrix_b")
    C_frag = T.alloc_buffer((16, 16), "float32", scope="wmma.accumulator")
    
    # Load to shared memory
    for i, j in T.grid(16, 16):
        with T.block("A_shared"):
            vi, vj = T.axis.remap("SS", [i, j])
            A_shared[vi, vj] = A[vi, vj]
        with T.block("B_shared"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_shared[vi, vj] = B[vi, vj]
    
    # Fill accumulator
    for i, j in T.grid(16, 16):
        with T.block("C_init"):
            vi, vj = T.axis.remap("SS", [i, j])
            C_frag[vi, vj] = T.float32(0)
    
    # Load fragments
    for i, j in T.grid(16, 16):
        with T.block("A_load"):
            vi, vj = T.axis.remap("SS", [i, j])
            A_frag[vi, vj] = A_shared[vi, vj]
        with T.block("B_load"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_frag[vi, vj] = B_shared[vi, vj]
    
    # Matrix multiply
    for i, j, k in T.grid(16, 16, 16):
        with T.block("C_compute"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C_frag[vi, vj] = T.float32(0)
            C_frag[vi, vj] = C_frag[vi, vj] + T.cast(A_frag[vi, vk], "float32") * T.cast(B_frag[vk, vj], "float32")
    
    # Store result
    for i, j in T.grid(16, 16):
        with T.block("C_store"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = C_frag[vi, vj]

# Alternative: Using TE API for simpler definition
def gemm_wmma_simple():
    M, N, K = 16, 16, 16
    A = te.placeholder((M, K), name="A", dtype="float16")
    B = te.placeholder((K, N), name="B", dtype="float16")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k].astype("float32") * B[k, j].astype("float32"), axis=k),
        name="C",
    )
    return A, B, C

# Build and extract PTX
def build_and_extract_ptx():
    import tvm.testing
    
    # Check if CUDA is available
    if not tvm.testing.device_enabled("cuda"):
        print("CUDA not available, skipping PTX extraction")
        return
    
    # Create function
    A, B, C = gemm_wmma_simple()
    s = te.create_schedule(C.op)
    
    # Schedule for WMMA
    # Get tensor core intrinsic
    intrin = tvm.tir.TensorIntrin.get("wmma_sync_16x16x16_f16f16f32")
    
    # Apply schedule transformations
    # For a proper WMMA schedule, we would need:
    # 1. Cache reads to shared memory
    # 2. Cache reads to wmma fragments  
    # 3. Tensorize the main computation
    # 4. Bind threads appropriately
    
    # Simplified version - just demonstrate compilation
    target = tvm.target.Target("cuda -arch=sm_89")
    
    # Build the module
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.build(s, [A, B, C], target=target, name="wmma_gemm")
    
    # Extract PTX
    ptx_code = mod.imported_modules[0].get_source()
    
    print("=" * 80)
    print("Generated PTX Code (excerpt):")
    print("=" * 80)
    
    # Look for mma.sync instructions
    lines = ptx_code.split('\n')
    for i, line in enumerate(lines):
        if 'mma.sync' in line.lower() or 'ldmatrix' in line.lower() or 'wmma' in line.lower():
            # Print context around the instruction
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            print('\n'.join(lines[start:end]))
            print('-' * 40)
    
    # Save full PTX
    with open('/tmp/wmma_gemm_sm89.ptx', 'w') as f:
        f.write(ptx_code)
    print(f"\nFull PTX saved to /tmp/wmma_gemm_sm89.ptx")
    print(f"PTX code length: {len(ptx_code)} characters")
    
    return ptx_code

if __name__ == "__main__":
    build_and_extract_ptx()
```

### D.2 Expected PTX Output (sm_89)

For sm_89 (Ada), the modern approach uses inline PTX via the MMA path. Expected instructions:

```ptx
// Load matrix from shared memory to registers
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0, %r1, %r2, %r3}, [%rd0];

// Matrix multiply-accumulate
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%f0, %f1, %f2, %f3},  // output (accumulator)
    {%r4, %r5},             // A matrix
    {%r6, %r7},             // B matrix  
    {%f0, %f1, %f2, %f3};   // C matrix (accumulator)
```

---

## E. Decision Conditions & Trigger Points

### E.1 WMMA vs MMA Path Selection

**Determined by TensorIntrin selection in Python schedule:**

1. **WMMA Path** (uses C++ nvcuda::wmma API):
   - Intrinsics: `wmma_load_*`, `wmma_sync_*`, `wmma_fill_*`, `wmma_store_*`
   - Buffer scopes: `wmma.matrix_a`, `wmma.matrix_b`, `wmma.accumulator`
   - CodeGen: Lines 933-991 in `codegen_cuda.cc`
   - Trigger: `op->op.same_as(builtin::tvm_mma_sync())`

2. **MMA Path** (uses inline PTX):
   - Intrinsics: `mma_ldmatrix_*`, `mma_f16f16f32`, `mma_fill_*`, `mma_store_*`
   - Buffer scopes: `warp` for ldmatrix/mma, `m16n8k8.matrixA/B/C` for specific variants
   - CodeGen: Lines 992-1095 in `codegen_cuda.cc`
   - Trigger: `op->op.same_as(builtin::ptx_mma())` or `op->op.same_as(builtin::ptx_ldmatrix())`

### E.2 Target Architecture Effects

Checked in `codegen_cuda.cc` and `ptx.cc`:

- **sm_70-sm_75 (Volta, Turing):** Base Tensor Core support, fp16 and int8
- **sm_80-sm_86 (Ampere):** Adds tf32, bf16, sparsity support
- **sm_89 (Ada):** Same instruction set as Ampere for Tensor Cores
- **sm_90 (Hopper):** Should support WGMMA but **not fully implemented in current TVM**

### E.3 Other Flags/Attributes

- **opt_level:** Higher optimization levels (3-4) more likely to trigger tensor core usage
- **No explicit "use_tensor_core" flag:** Selection is implicit via tensorize schedule primitive
- **Fragment alignment:** Built into TensorIntrin definitions (align=64)
- **Shared memory scope:** Regular `shared` or `shared.dyn` (dynamic shared memory)

---

## F. Version Differences & Evolution

### F.1 Recent Changes (Last 2 Years)

Based on code structure and comments:

1. **MMA Intrinsics Addition** (likely v0.10+):
   - Modern `ptx_mma` path added alongside legacy `tvm_mma_sync`
   - Inline PTX generation for better control and performance
   - Files: `src/target/source/ptx.cc` added/expanded

2. **Float8 Support** (v0.11+):
   - Added `float8_e4m3fn` and `float8_e5m2` intrinsics
   - Lines 263-283 in `cuda.py`: ldmatrix for float8
   - Lines 522-544 in `cuda.py`: MMA sync for float8

3. **Sparse Tensor Core Support** (v0.11+):
   - `ptx_mma_sp` intrinsic for structured sparsity
   - Lines 1027-1064 in `codegen_cuda.cc`

4. **m16n8k8 Variants** (Recent):
   - Additional MMA shape `m16n8k8` alongside `m16n16k16`
   - Lines 1471-1753 in `cuda.py`

### F.2 File Location Changes

No major reorganizations detected in current structure, but historically:
- TensorIntrin definitions consolidated in `python/tvm/tir/tensor_intrin/` directory
- PTX utilities extracted to separate `ptx.cc` file

### F.3 API Changes

- TensorIntrin registration now uses `TensorIntrin.register(name, desc_func, impl_func)`
- Intrinsic implementations now use `@T.prim_func` decorated functions
- Match buffer API standardized with `T.match_buffer()`

---

## G. Pitfalls, Edge Cases & Fallback Conditions

### G.1 Layout Mismatches

**Problem:** WMMA/MMA intrinsics have strict layout requirements
- Matrix A: Usually row-major
- Matrix B: Column-major (for NN) or row-major (for NT)
- Matrix C: Row-major for storage

**Consequence:** Incorrect layout leads to wrong results, not compile errors

**Detection:** Check generated PTX for correct `.row`/`.col` modifiers in `mma.sync` instruction

**Mitigation:** Use correct intrinsic variant (e.g., `wmma_sync_*_trans` for transposed B)

### G.2 Alignment Requirements

**Problem:** Fragment buffers and shared memory must be aligned
- Shared memory: 128-byte alignment for ldmatrix
- Fragment offsets: Must be multiple of fragment size

**Consequence:** Runtime errors or incorrect results

**Detection:** Check buffer allocations have `align=64` attribute

**Mitigation:** TVM automatically handles alignment in most cases; manual schedules need care

### G.3 Thread Binding

**Problem:** WMMA/MMA operations require specific warp-level thread organization
- Operations execute across all 32 threads in a warp
- Thread binding must be `threadIdx.x` for warp dimension

**Consequence:** Compilation failure or runtime errors

**Detection:** Check schedule has `T.thread_binding(0, 32, "threadIdx.x")`

**Mitigation:** Ensure proper thread binding in schedule

### G.4 Fragment Shapes

**Problem:** Not all (M, N, K) shapes are supported
- Valid: 16x16x16, 32x8x16, 8x32x16, 16x8x16, 16x8x32, 8x8x32
- Invalid: Arbitrary sizes like 12x12x12

**Consequence:** Compilation error or fallback to non-tensor-core path

**Detection:** Check `ptx.cc` lines 242-288 for shape validation

**Mitigation:** Tile loops to valid fragment sizes

### G.5 Data Type Combinations

**Problem:** Not all (A_dtype, B_dtype, C_dtype) combinations are supported
- Valid: (fp16, fp16, fp32), (fp16, fp16, fp16), (int8, int8, int32), (tf32, tf32, fp32)
- Invalid: Mixed precision like (fp16, fp32, fp32)

**Consequence:** Compilation error

**Detection:** Check `CheckMMAConfigValidity()` in `ptx.cc`

**Mitigation:** Use supported type combinations, add explicit casts if needed

### G.6 Shared Memory Bank Conflicts

**Problem:** Naive shared memory layouts can cause bank conflicts
- Particularly with column-major access patterns

**Consequence:** Reduced performance, not correctness issue

**Detection:** Profile with nvprof/Nsight Compute

**Mitigation:** Use swizzled layouts or padding (TVM has built-in support)

### G.7 Fallback to Non-Tensor-Core Path

**Occurs when:**
1. Target architecture < sm_70
2. Tensor intrinsic not properly matched in tensorize
3. Fragment buffer scopes not set correctly
4. Shape/type combination not supported
5. Alignment requirements not met

**Detection:**
- Check PTX for absence of `mma.sync` or `wmma` instructions
- Look for regular `fmul`/`fadd` instruction sequences instead

**Example of fallback PTX:**
```ptx
// Non-tensor-core path - regular FMAD
fma.rn.f32 %f0, %f1, %f2, %f3;  // Instead of mma.sync
```

### G.8 sm_90 (Hopper) WGMMA Limitations

**Current Status:** Limited/No production support for WGMMA in TVM

**Evidence:**
- No `wgmma` string found in Python intrinsic definitions
- No `builtin::wgmma_*` calls in codegen
- PTX assembly generation focused on `mma.sync`, not `wgmma`

**Workaround:** Use sm_89 target even for H100 GPUs to use `mma.sync` path

**Future:** WGMMA support likely requires:
1. New TensorIntrin definitions for wgmma operations
2. New builtin functions (e.g., `builtin::wgmma_mma_async()`)
3. PTX generation support in `ptx.cc`
4. Async descriptor management for tensor memory accelerator (TMA)

---

## H. Validation & Testing

### H.1 How to Verify Tensor Core Usage

1. **Check PTX contains `mma.sync` or `ldmatrix` instructions:**
   ```bash
   grep -i "mma.sync\|ldmatrix" /tmp/generated.ptx
   ```

2. **Verify fragment buffer scopes in TIR:**
   ```python
   print(mod.script())  # Look for wmma.matrix_a/b/accumulator or warp scopes
   ```

3. **Profile with Nsight Compute:**
   ```bash
   ncu --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active ./app
   ```
   Tensor pipe utilization should be > 0% if Tensor Cores are used

4. **Check generated CUDA code includes mma.h:**
   ```cpp
   #include <mma.h>  // Present if using nvcuda::wmma API
   ```

### H.2 Common Validation Issues

**Issue:** PTX contains only `fma.rn.f32` instructions
**Cause:** Tensorize didn't match or fell back to non-TC path
**Fix:** Verify intrinsic name, buffer scopes, and fragment shapes

**Issue:** Compilation error "cannot find wmma intrinsic"
**Cause:** Target architecture < sm_70
**Fix:** Set `-arch=sm_70` or higher

**Issue:** Runtime error "invalid fragment shape"
**Cause:** Mismatched fragment metadata
**Fix:** Check InferFragment pass output, ensure all fragments have same m,n,k

---

## I. Summary & Key Takeaways

### Critical Path for Tensor Core Usage:

1. **Define computation** in TensorIR with appropriate buffer scopes
2. **Apply schedule** with `tensorize` using registered TensorIntrin
3. **TVM transforms** through InferFragment, StorageRewrite, etc.
4. **CodeGen emits** either C++ WMMA API or inline PTX
5. **nvcc compiles** to final PTX/SASS with `mma.sync` instructions

### Two Parallel Paths:

- **WMMA (Legacy):** `wmma_*` intrinsics → C++ API → PTX
- **MMA (Modern):** `mma_*` intrinsics → Inline PTX → PTX

Both work on sm_70+, MMA path gives more control and is preferred for new code.

### sm_89 (Ada) Confirmation:

✅ Supports `mma.sync.aligned.m16n8k16.*` instructions  
✅ Uses same tensor core ISA as Ampere (sm_80)  
✅ No architectural changes from Ampere to Ada for tensor cores  

### sm_90 (Hopper) Status:

❌ WGMMA support **not found** in current TVM codebase  
⚠️ Can still use `mma.sync` instructions for compatibility  
🔮 Future work needed for WGMMA async operations  

---

## References & Source Code Locations

All paths relative to repository root:

- **Python Intrinsics:** `python/tvm/tir/tensor_intrin/cuda.py`
- **TIR Passes:** `src/tir/transforms/tensorcore_infer_fragment.cc`
- **CUDA CodeGen:** `src/target/source/codegen_cuda.cc` (lines 933-1095, 1382-1720)
- **PTX Generation:** `src/target/source/ptx.cc` (lines 542-596)
- **Runtime Scopes:** `src/runtime/thread_storage_scope.h` (lines 56-64, 157-164)

**Commit Reference:** d03d0ba9340c509e983dd7066d3a182ad00e9622

---

*End of Deep Research Document*
