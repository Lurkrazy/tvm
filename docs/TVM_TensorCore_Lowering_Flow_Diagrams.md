# TVM TIR to Tensor Core Lowering - Visual Flow Diagrams

## 1. High-Level Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                        USER APPLICATION LEVEL                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Python Code:                                                          │
│    A = te.placeholder((M, K), dtype="float16")                        │
│    B = te.placeholder((K, N), dtype="float16")                        │
│    C = te.compute(...)  # Matrix multiplication                       │
│                                                                         │
│    s = te.create_schedule(C.op)                                       │
│    s.tensorize(block, "wmma_sync_16x16x16_f16f16f32")  # ← KEY STEP   │
│                                                                         │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         │ TVM Build Process
                         ↓
┌────────────────────────────────────────────────────────────────────────┐
│                    TENSOR INTRIN REGISTRATION                          │
│                 (python/tvm/tir/tensor_intrin/cuda.py)                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TensorIntrin.register(                                                │
│      "wmma_sync_16x16x16_f16f16f32",                                  │
│      desc_func,   # High-level computation description                │
│      impl_func    # Implementation with tvm_mma_sync/ptx_mma calls    │
│  )                                                                     │
│                                                                         │
│  Intrinsic Types:                                                      │
│  ┌──────────────────┬────────────────────────────────────────────┐   │
│  │ WMMA Path        │ MMA Path (Modern)                          │   │
│  ├──────────────────┼────────────────────────────────────────────┤   │
│  │ wmma_load_*      │ mma_ldmatrix_*                            │   │
│  │ wmma_sync_*      │ mma_f16f16f32, mma_i8i8i32, ...           │   │
│  │ wmma_fill_*      │ mma_fill_*                                │   │
│  │ wmma_store_*     │ mma_store_*                               │   │
│  │                  │                                            │   │
│  │ Buffer scopes:   │ Buffer scopes:                            │   │
│  │ wmma.matrix_a    │ warp                                      │   │
│  │ wmma.matrix_b    │ m16n8k8.matrixA/B/C                       │   │
│  │ wmma.accumulator │                                            │   │
│  └──────────────────┴────────────────────────────────────────────┘   │
│                                                                         │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         │ Schedule Application
                         ↓
┌────────────────────────────────────────────────────────────────────────┐
│                          TIR GENERATION                                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Generated TIR contains intrinsic calls:                               │
│                                                                         │
│  WMMA Path Example:                                                    │
│    T.tvm_load_matrix_sync(buf, 16, 16, 16, idx, ptr, stride, layout) │
│    T.tvm_mma_sync(d, d_idx, a, a_idx, b, b_idx, c, c_idx)            │
│    T.tvm_store_matrix_sync(buf, 16, 16, 16, idx, ptr, stride, "row") │
│                                                                         │
│  MMA Path Example:                                                     │
│    T.ptx_ldmatrix(trans, 4, ".b16", warp_buf, offset, smem, stride)  │
│    T.ptx_mma("m16n8k16", "row", "col", "fp16", ...)                  │
│                                                                         │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         │ TIR Pass Pipeline
                         ↓
┌────────────────────────────────────────────────────────────────────────┐
│                    TIR TRANSFORMATION PASSES                           │
│                     (src/tir/transforms/*.cc)                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Pass Execution Order:                                                 │
│                                                                         │
│  1. StorageRewrite                                                     │
│     └─► Allocate fragment buffers with correct scopes                 │
│                                                                         │
│  2. InferFragment                                                      │
│     └─► Extract (m, n, k, layout) from intrinsic calls                │
│     └─► Add fragment_shape and fragment_layout attributes             │
│                                                                         │
│  3. MemhammerTensorCoreRewrite (if using memhammer)                   │
│     └─► Rewrite data copies to use wmma::load_matrix_sync            │
│                                                                         │
│  4. LowerWarpMemory                                                    │
│     └─► Handle warp-level memory transformations                      │
│                                                                         │
│  5. LowerIntrin                                                        │
│     └─► Lower platform-agnostic intrinsics                            │
│                                                                         │
│  6. SplitHostDevice                                                    │
│     └─► Separate host and device code                                 │
│                                                                         │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         │ Code Generation
                         ↓
┌────────────────────────────────────────────────────────────────────────┐
│                         CODEGEN LAYER (C++)                            │
│              (src/target/source/codegen_cuda.cc, ptx.cc)              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CodeGenCUDA::VisitExpr_(const CallNode* op)                          │
│                                                                         │
│  Pattern Matching:                                                     │
│                                                                         │
│  ┌──────────────────────────────┬────────────────────────────────────┐ │
│  │ if (op->op.same_as(...))     │ Generated Code                    │ │
│  ├──────────────────────────────┼────────────────────────────────────┤ │
│  │ builtin::tvm_fill_fragment   │ nvcuda::wmma::fill_fragment(...)  │ │
│  │                              │ [Lines 933-942]                   │ │
│  ├──────────────────────────────┼────────────────────────────────────┤ │
│  │ builtin::tvm_load_matrix     │ nvcuda::wmma::load_matrix_sync()  │ │
│  │                              │ [Lines 943-954]                   │ │
│  ├──────────────────────────────┼────────────────────────────────────┤ │
│  │ builtin::tvm_mma_sync        │ nvcuda::wmma::mma_sync(...)       │ │
│  │                              │ [Lines 972-981]                   │ │
│  ├──────────────────────────────┼────────────────────────────────────┤ │
│  │ builtin::ptx_mma             │ PrintMMAAssembly()                │ │
│  │                              │ → inline PTX asm                  │ │
│  │                              │ [Lines 992-1026]                  │ │
│  ├──────────────────────────────┼────────────────────────────────────┤ │
│  │ builtin::ptx_ldmatrix        │ PrintLoadMatrixAssembly()         │ │
│  │                              │ → inline PTX asm                  │ │
│  │                              │ [Lines 1065-1095]                 │ │
│  └──────────────────────────────┴────────────────────────────────────┘ │
│                                                                         │
│  Fragment Type Declaration:                                            │
│    if (scope == "wmma.matrix_a")                                      │
│      → nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,    │
│                                 half, nvcuda::wmma::row_major>        │
│    [Lines 1382-1720]                                                  │
│                                                                         │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         │ CUDA/PTX Source Generation
                         ↓
┌────────────────────────────────────────────────────────────────────────┐
│                         GENERATED CUDA/PTX                             │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Option A: C++ WMMA API (from tvm_mma_sync intrinsics)                │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ #include <mma.h>                                                │   │
│  │                                                                  │   │
│  │ nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,     │   │
│  │                        half, nvcuda::wmma::row_major> a_frag;   │   │
│  │                                                                  │   │
│  │ nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, lda);            │   │
│  │ nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, ldb);            │   │
│  │ nvcuda::wmma::fill_fragment(c_frag, 0.0f);                     │   │
│  │                                                                  │   │
│  │ nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);        │   │
│  │                                                                  │   │
│  │ nvcuda::wmma::store_matrix_sync(c_ptr, c_frag, ldc,            │   │
│  │                                  nvcuda::wmma::mem_row_major);  │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Option B: Inline PTX (from ptx_mma intrinsics)                       │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ __asm__ __volatile__(                                           │   │
│  │   "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "                  │   │
│  │   "{%0, %1, %2, %3}, [%4];\n"                                  │   │
│  │   : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])            │   │
│  │   : "r"(addr_a));                                               │   │
│  │                                                                  │   │
│  │ __asm__ __volatile__(                                           │   │
│  │   "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "         │   │
│  │   "{%0, %1, %2, %3}, "                                         │   │
│  │   "{%4, %5}, "                                                  │   │
│  │   "{%6, %7}, "                                                  │   │
│  │   "{%0, %1, %2, %3};\n"                                        │   │
│  │   : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])            │   │
│  │   : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));              │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         │ NVCC Compilation
                         ↓
┌────────────────────────────────────────────────────────────────────────┐
│                          FINAL PTX ASSEMBLY                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  .version 7.0                                                          │
│  .target sm_89                                                         │
│  .address_size 64                                                      │
│                                                                         │
│  .visible .entry wmma_gemm_kernel(...) {                              │
│    .reg .pred %p<8>;                                                  │
│    .reg .f16 %h<32>;                                                  │
│    .reg .f32 %f<32>;                                                  │
│    .reg .b32 %r<64>;                                                  │
│                                                                         │
│    // Load matrices from shared memory                                │
│    ldmatrix.sync.aligned.m8n8.x4.shared.b16                          │
│        {%r0, %r1, %r2, %r3}, [%rd0];                                 │
│                                                                         │
│    // Tensor Core matrix multiply-accumulate                          │
│    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32                 │
│        {%f0, %f1, %f2, %f3},      // D (output)                      │
│        {%r4, %r5},                 // A matrix                        │
│        {%r6, %r7},                 // B matrix                        │
│        {%f0, %f1, %f2, %f3};       // C (accumulator)                │
│                                                                         │
│    // Store result to global memory                                   │
│    st.global.f32 [%rd1], %f0;                                         │
│    ...                                                                 │
│  }                                                                     │
│                                                                         │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         │ PTX → SASS
                         ↓
┌────────────────────────────────────────────────────────────────────────┐
│                       NATIVE GPU MACHINE CODE                          │
│                               (SASS)                                   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Binary instructions executed directly by GPU                          │
│  - Tensor Core instructions (HMMA, IMMA, etc.)                        │
│  - Memory operations (LDG, STG, LDS, STS)                             │
│  - Control flow (BRA, SSY, etc.)                                      │
│                                                                         │
│  Executed on Tensor Cores within each SM                              │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## 2. Decision Tree for Path Selection

```
                     User calls tensorize(block, intrin_name)
                                    │
                                    ↓
                    ┌───────────────────────────────┐
                    │  What intrinsic name?         │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ↓                                ↓
          ┌──────────────────┐            ┌──────────────────┐
          │ Starts with       │            │ Starts with      │
          │ "wmma_*"?         │            │ "mma_*"?         │
          └────────┬──────────┘            └────────┬─────────┘
                   │ YES                             │ YES
                   ↓                                 ↓
          ┌──────────────────┐            ┌──────────────────┐
          │   WMMA PATH      │            │    MMA PATH      │
          │   (C++ API)      │            │  (Inline PTX)    │
          └────────┬─────────┘            └────────┬─────────┘
                   │                                │
                   ↓                                ↓
          ┌──────────────────┐            ┌──────────────────┐
          │ Buffer Scopes:   │            │ Buffer Scopes:   │
          │ • wmma.matrix_a  │            │ • warp           │
          │ • wmma.matrix_b  │            │ • m16n8k8.*      │
          │ • wmma.accumulator│           │                  │
          └────────┬─────────┘            └────────┬─────────┘
                   │                                │
                   ↓                                ↓
          ┌──────────────────┐            ┌──────────────────┐
          │ TIR Builtins:    │            │ TIR Builtins:    │
          │ • tvm_load_matrix│            │ • ptx_ldmatrix   │
          │ • tvm_mma_sync   │            │ • ptx_mma        │
          │ • tvm_fill_frag  │            │ • mma_fill       │
          │ • tvm_store_matrix│           │ • mma_store      │
          └────────┬─────────┘            └────────┬─────────┘
                   │                                │
                   ↓                                ↓
          ┌──────────────────┐            ┌──────────────────┐
          │ CodeGen emits:   │            │ CodeGen emits:   │
          │ nvcuda::wmma::   │            │ __asm__ volatile │
          │ • load_matrix_   │            │ ("mma.sync.      │
          │   sync()         │            │   aligned...")    │
          │ • mma_sync()     │            │                  │
          │ • fill_fragment()│            │                  │
          │ • store_matrix_  │            │                  │
          │   sync()         │            │                  │
          └────────┬─────────┘            └────────┬─────────┘
                   │                                │
                   └────────────┬───────────────────┘
                                │
                                ↓
                        ┌───────────────────┐
                        │  NVCC compiles    │
                        │  to PTX/SASS with │
                        │  Tensor Core      │
                        │  instructions     │
                        └───────────────────┘
```

## 3. Fragment Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                          GLOBAL MEMORY                              │
│                                                                      │
│  Input Matrices A, B (float16)                                     │
│  Output Matrix C (float32)                                         │
│  Size: M×K, K×N, M×N                                              │
│                                                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Load (global → shared)
                               │ cp.async or regular ld.global
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      SHARED MEMORY (Per Block)                      │
│                                                                      │
│  Cached Tiles: A_shared, B_shared                                  │
│  Size: Typically 16×16 or larger tiles                             │
│  Scope: "shared" or "shared.dyn"                                   │
│  Alignment: 128-byte for ldmatrix                                  │
│                                                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Load (shared → warp fragments)
                               │ ldmatrix.sync or wmma::load_matrix_sync
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    WARP-LEVEL FRAGMENTS (Registers)                 │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  WMMA Path                   │  MMA Path                      │  │
│  ├──────────────────────────────┼────────────────────────────────┤  │
│  │ Scope: wmma.matrix_a         │ Scope: warp                    │  │
│  │ Type: nvcuda::wmma::fragment │ Type: uint32_t array           │  │
│  │ Size: 16×16 (8 registers)    │ Size: varies by shape          │  │
│  │                              │                                │  │
│  │ A_frag: matrix_a fragment    │ A_warp: ldmatrix loaded data  │  │
│  │ B_frag: matrix_b fragment    │ B_warp: ldmatrix loaded data  │  │
│  │ C_frag: accumulator fragment │ C_warp: accumulator registers │  │
│  └──────────────────────────────┴────────────────────────────────┘  │
│                                                                      │
│  Each warp (32 threads) collectively holds one fragment             │
│  Data distributed across thread registers                           │
│                                                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Compute
                               │ mma.sync or wmma::mma_sync
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      TENSOR CORE EXECUTION                          │
│                                                                      │
│  Matrix Multiply-Accumulate: D = A × B + C                         │
│  Executed on dedicated Tensor Core units                           │
│  Per warp: 16×16×16 (or other supported shapes)                   │
│  Throughput: Much higher than CUDA cores for matrix ops            │
│                                                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Store (fragments → shared/global)
                               │ wmma::store_matrix_sync or manual
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   RESULT WRITEBACK TO MEMORY                        │
│                                                                      │
│  C_frag → Shared (optional, for further processing)                │
│  C_frag → Global (final output)                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 4. Supported Matrix Shapes and Types

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   TENSOR CORE SHAPE SUPPORT MATRIX                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Shape      │ Input Types  │ Output Type │ Architecture │ Path           │
│  ───────────┼──────────────┼─────────────┼──────────────┼────────────── │
│  m16n16k16  │ fp16 × fp16  │ fp16/fp32   │ sm_70+       │ WMMA & MMA    │
│  m16n16k16  │ int8 × int8  │ int32       │ sm_75+       │ WMMA          │
│  m16n16k16  │ int4 × int4  │ int32       │ sm_75+       │ WMMA          │
│  m16n8k16   │ fp16 × fp16  │ fp16/fp32   │ sm_70+       │ MMA           │
│  m16n8k32   │ int8 × int8  │ int32       │ sm_75+       │ MMA           │
│  m8n8k32    │ int4 × int4  │ int32       │ sm_75+       │ WMMA          │
│  m16n16k8   │ tf32 × tf32  │ fp32        │ sm_80+       │ MMA           │
│  m16n16k16  │ bf16 × bf16  │ fp32        │ sm_80+       │ MMA           │
│  m16n8k32   │ fp8 × fp8    │ fp32        │ sm_89+       │ MMA           │
│                                                                           │
│  Layout Variations:                                                      │
│    • A: row_major or col_major                                          │
│    • B: row_major or col_major                                          │
│    • C: always row_major for storage                                    │
│                                                                           │
│  TVM Intrinsic Naming Convention:                                       │
│    wmma_sync_<shape>_<A_type><B_type><C_type>[_trans]                  │
│    mma_<A_type><B_type><C_type>[_trans_a][_trans_b]                    │
│                                                                           │
│  Examples:                                                               │
│    • wmma_sync_16x16x16_f16f16f32        (16×16×16, fp16→fp32, NN)     │
│    • wmma_sync_16x16x16_f16f16f32_trans  (16×16×16, fp16→fp32, NT)     │
│    • mma_f16f16f32                        (16×8×16, fp16→fp32, NN)      │
│    • mma_i8i8i32_trans_b                  (16×8×32, int8→int32, NT)     │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

## 5. Validation Checklist

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  TENSOR CORE USAGE VALIDATION CHECKLIST                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Phase 1: Pre-Build Checks                                              │
│  ───────────────────────────────────────────────────────────────────    │
│  [ ] Target architecture >= sm_70                                        │
│  [ ] Intrinsic name matches registered TensorIntrin                     │
│  [ ] Buffer scopes correctly set (wmma.* or warp)                       │
│  [ ] Fragment shapes are valid (16×16×16, etc.)                         │
│  [ ] Data type combination supported                                     │
│  [ ] Layout specification correct (row/col major)                        │
│  [ ] Thread binding present (threadIdx.x)                               │
│                                                                           │
│  Phase 2: TIR Inspection                                                │
│  ───────────────────────────────────────────────────────────────────    │
│  [ ] Print mod.script() and verify:                                     │
│      • Correct buffer scopes present                                    │
│      • Intrinsic calls visible (T.tvm_mma_sync or T.ptx_mma)           │
│      • Fragment allocations have proper alignment                       │
│      • Load/store operations present                                    │
│                                                                           │
│  Phase 3: PTX Analysis                                                  │
│  ───────────────────────────────────────────────────────────────────    │
│  [ ] Extract PTX: mod.imported_modules[0].get_source()                  │
│  [ ] Search for:                                                         │
│      • "mma.sync.aligned" instructions                                  │
│      • "ldmatrix.sync" instructions                                     │
│      • "nvcuda::wmma::" API calls                                       │
│  [ ] Verify absence of excessive fma.rn.f32 (fallback indicator)       │
│                                                                           │
│  Phase 4: Runtime Profiling                                             │
│  ───────────────────────────────────────────────────────────────────    │
│  [ ] Use Nsight Compute:                                                │
│      ncu --metrics sm__inst_executed_pipe_tensor.avg ./app              │
│  [ ] Verify Tensor Core utilization > 0%                               │
│  [ ] Check memory bandwidth (should be high for Tensor Cores)           │
│  [ ] Validate performance vs baseline                                   │
│                                                                           │
│  Phase 5: Correctness Testing                                           │
│  ───────────────────────────────────────────────────────────────────    │
│  [ ] Run against reference implementation                               │
│  [ ] Check numerical accuracy (rtol ~1e-2 for fp16)                    │
│  [ ] Test edge cases (boundary conditions)                              │
│  [ ] Verify deterministic results                                       │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

*These diagrams complement the main research document in:*  
*`docs/TVM_TIR_to_TensorCore_Lowering_Deep_Research.md`*
