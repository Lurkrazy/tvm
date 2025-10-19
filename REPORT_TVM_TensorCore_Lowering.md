## Executive Summary
- **结论**：TVM 在 CUDA 目标上通过已注册的 TensorIntrin（`wmma_*`/`mma_*` + `ldmatrix`）配合 `tensorize` 将 TensorIR 的块/循环重写为 TIR 内建调用：`tir.tvm_load_matrix_sync`、`tir.tvm_mma_sync`、`tir.tvm_store_matrix_sync`、`tir.ptx_ldmatrix`。在默认 TIR 编译管线中，经 `InferFragment` 附加 fragment 元数据，最后在 CUDA 源码后端将这些调用映射到 C++ WMMA API（`nvcuda::wmma::*`），以及必要时的 **inline PTX**（`mma.sync.aligned.*`、`ldmatrix.sync.aligned.*`）。
- **触发条件**：目标 `target="cuda -arch=sm_89"` 且调度使用了 WMMA/ldmatrix 的 TensorIntrin；A 选 `wmma.matrix_a` 行主，B 选 `wmma.matrix_b` 列主（或使用 `*_trans` 变体）；缓冲区 scope/对齐/offset_factor 與碎片形状匹配。Hopper `sm_90` 的 `wgmma`：本仓库版本未发现 `wgmma` 支持（源码无 `wgmma` 标识），因此不会触发 `wgmma.mma_async.*`。

---

## 架构图
```mermaid
flowchart TD
  A[TensorIR Schedule\n- cache_read/write 到 shared/warp/wmma scopes\n- transform_layout\n- tensorize(wmma/ldmatrix)]
  B[TIR with Intrin Calls\n- tir.tvm_load_matrix_sync\n- tir.tvm_mma_sync / tvm_bmma_sync\n- tir.tvm_store_matrix_sync\n- tir.ptx_ldmatrix]
  C[Passes (device)]
  D[CodeGen CUDA]
  E[NVRTC/ptx]
  F[PTX/SASS]

  A --> B
  B --> C
  C -->|InferFragment / LowerWarpMemory / LowerIntrin| D
  D -->|nvcuda::wmma::* / inline PTX| E
  E -->|mma.sync.aligned.*, ldmatrix.sync.aligned.*| F
```

---

## 源码清单（commit: `c2144d4bfdde48790d968f4d61ff8d782381eb60`）
- `python/tvm/tir/tensor_intrin/cuda.py` :: 注册并生成 CUDA TensorCore 相关 TensorIntrin（WMMA、MMA、LDMATRIX）及签名，`tensorize` 绑定使用 :: `get_wmma_*_intrin`, `get_ldmatrix_intrin`, `TensorIntrin.register(...)` :: 定义 `wmma.matrix_a/b/accumulator`、形状/布局/对齐。
- `src/tir/op/builtin.cc` :: 定义 TIR 内建 call 节点 :: `tvm_load_matrix_sync`, `tvm_mma_sync`, `tvm_store_matrix_sync`, `ptx_ldmatrix`, `ptx_mma`, `ptx_mma_sp`。
- `python/tvm/tir/pipeline.py` :: 默认 TIR 管线与终结 device/host passes 顺序 :: `InferFragment`, `LowerWarpMemory`, `LowerIntrin` 等加入设备侧；`LowerTVMBuiltin`, `LowerIntrin` 等加入主机侧。
- `src/tir/transforms/tensorcore_infer_fragment.cc` :: 从内建调用/fragment 分配中推断并写入 fragment 元数据与一致性校验 :: `InferFragment()` pass。
- `src/target/source/codegen_cuda.cc` :: 将上述内建映射到 WMMA C++/inline PTX/ldmatrix inline PTX；在需要时注入 `#include <mma.h>` :: 分支处理 `tvm_*`、`ptx_*`，并生成 `nvcuda::wmma::*` 或 `mma.sync.aligned.*`、`ldmatrix.sync.aligned.*`。
- `src/target/source/ptx.cc` :: 生成 `mma.sync.aligned.*` 与 `ldmatrix.sync.aligned.*` 的 inline PTX 模板与操作数拼接 :: `PrintMMAAssembly`, `PrintLoadMatrixAssembly`。
- `src/runtime/thread_storage_scope.h` :: `wmma.matrix_a/b/accumulator` 等存储 scope 常量与解析。
- `python/tvm/testing/tir.py`、`python/tvm/dlight/gpu/matmul.py`、`src/meta_schedule/schedule_rule/multi_level_tiling_tensor_core.cc` :: 自动/模板化调度中如何选取 `wmma` intrin 组并应用到 matmul。
- `src/target/opt/build_cuda_on.cc` :: 使用 NVRTC 将 C++ CUDA 源编译为 PTX；由 `CodeGenCUDA` 生成 WMMA API/inline PTX 源，并经 NVRTC 得到 PTX。

---

## 显式映射表（WMMA 16x16x16 FP16）
- **wmma.load**
  - TIR: `tir.tvm_load_matrix_sync(ptr, m,n,k, frag_idx, smem_ptr, stride, layout)`
  - CodeGen: `nvcuda::wmma::load_matrix_sync(frag[frag_idx], smem_ptr, stride)` 或 `ldmatrix.sync.aligned.*`（经 `tir.ptx_ldmatrix`）
  - 约束: A: `wmma.matrix_a` row_major；B: `wmma.matrix_b` col_major（或使用 trans 变体）；对齐 64、`offset_factor` = 碎片 n 维；共享内存。
- **wmma.mma**
  - TIR: `tir.tvm_mma_sync(dptr, didx, aptr, aidx, bptr, bidx, cptr, cidx)`
  - CodeGen: `nvcuda::wmma::mma_sync(d[i], a[i], b[i], c[i])`
  - PTX: 如果使用 `tir.ptx_mma` 路径，则直接生成 `mma.sync.aligned.m16n16k16.{layout}.{types}`（见 `src/target/source/ptx.cc:542+`）。
- **wmma.store**
  - TIR: `tir.tvm_store_matrix_sync(aptr, m,n,k, frag_idx, gmem_ptr, stride, layout)`
  - CodeGen: `nvcuda::wmma::store_matrix_sync(dst, frag[frag_idx], stride, nvcuda::wmma::mem_row_major)`
- **ldmatrix**
  - TIR: `tir.ptx_ldmatrix(trans, num, ".b16", warp_buf, warp_off, smem_ptr, smem_off)`
  - PTX: `ldmatrix.sync.aligned.m8n8.x{1|2|4}{.trans}.shared.b16`（见 `src/target/source/ptx.cc:619+`）。

形状/类型/布局：
- `wmma_sync_16x16x16_f16f16f32[_trans]`、`wmma_load_16x16x16_f16_a_*`、`wmma_load_16x16x16_f16_b[_trans]_*`；B trans 代表 col-major。

---

## 最小复现实验（MRE）
- 脚本：`mre_wmma_m16n16k16_fp16.py`（本仓库已写入）
- 用法：
  - 安装可用的 TVM(含 CUDA) 与 CUDA/NVRTC，执行：
    - `python3 mre_wmma_m16n16k16_fp16.py`
  - 目标：`target="cuda -arch=sm_89"`
- 期望 PTX 证据：输出中至少包含：
  - `mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32`（或 f16/f16/f32 组合）
  - `ldmatrix.sync.aligned.m8n8.x4.shared.b16`（可能带 `.trans`）

---

## TIR Pass 顺序（关键与 TensorCore 相关）
- `python/tvm/tir/pipeline.py` 中 device 部分：
  - `InferFragment()`（附加并校验 `wmma.*` fragment 形状/布局）
  - `LowerWarpMemory()`（warp scope 降低）
  - `LowerIntrin()`（目标相关内建降级）
  - 另：`InjectSoftwarePipeline` 可能引入 `cp.async.*`/commit_group/wait_group；`InjectPTXLDG32` 可选
- host 部分稍后执行 `LowerTVMBuiltin`、`LowerIntrin` 等。

---

## CodeGen 决策与触发条件
- 映射点：`src/target/source/codegen_cuda.cc`
  - `tvm_fill_fragment` → `nvcuda::wmma::fill_fragment`（L933-L943）
  - `tvm_load_matrix_sync` → `nvcuda::wmma::load_matrix_sync`（L944-L955）
  - `tvm_store_matrix_sync` → `nvcuda::wmma::store_matrix_sync`（L955-L972）
  - `tvm_mma_sync` → `nvcuda::wmma::mma_sync`（L972-L981）
  - `ptx_mma`/`ptx_mma_sp` → `mma.sync.aligned.*`/稀疏变体 inline PTX（L992-L1065）
  - `ptx_ldmatrix` → `ldmatrix.sync.aligned.*` inline PTX（L1065-L1095）
  - 触发 `#include <mma.h>`：设置 `need_mma_h_`（L290, 934, 944, 956, 973, 983, 1706+）
- NVPTX 后端：LLVM NVVM intrinsic 路径未在本版本直接使用 `llvm.nvvm.wmma.*`；TVM 采用 C++ WMMA API 与 inline PTX 组合。

---

## sm_90 / WGMMA 支持现状
- 检索结果：本仓库源码无 `wgmma` / `wgmma.mma_async` 标识（`rg` 无匹配），未见 Python 端 TensorIntrin 或 CodeGen 分支。因此：
  - **结论**：当前版本 TVM 不会生成 `wgmma.mma_async.*`。Hopper 上仍走 `mma.sync`/WMMA 或 CUTLASS 等外部路径。
  - 若需 WGMMA，需要参考后续上游 PR/issue（本报告未在源码中发现）。

---

## 版本差异（近两年要点）
- TensorIR/MetaSchedule/DLight 中逐步统一 WMMA/MMA/LDMATRIX 的 TensorIntrin 注册于 `python/tvm/tir/tensor_intrin/cuda.py`；`meta_schedule` 提供 `MultiLevelTilingTensorCore` 默认规则。
- CodeGen 仍以 WMMA C++ API + 内联 PTX 实现；NVVM wmma intrinsic 路径并未成为主路径。
- Hopper WGMMA：本版本无实现迹象（无符号、无 intrin、无 codegen）。

---

## 坑点与边界
- **布局**：A 一般 row-major → `wmma.matrix_a`；B 多为 col-major → `wmma.matrix_b` 或使用 `*_trans` 变体（TensorIntrin 名称含 `trans`）。
- **对齐/offset_factor**：TensorIntrin 要求 `align=64`、`offset_factor` = `frag_n` 等；不满足将无法匹配 tensorize 模式或在 `InferFragment`/`LowerIntrin` 报错。
- **碎片形状**：16x16x16（FP16/FP32 累加），int8 仍 16x16x16；另有 8x8x32 s4 等特殊形状。形状必须一致，见 `tensorcore_infer_fragment.cc` 校验。
- **ldmatrix 约束**：仅支持 `.b16`，且 `num` 为 1/2/4；当 `trans` 且 dtype 为 int8 会退化为手写循环（见 `codegen_cuda.cc:1081-1090`）。
- **退化路径**：未使用/匹配 TensorIntrin、scope 布局不符、目标非 CUDA 或 `-arch` 低于 7.0 → 不会触发 TensorCore；`python/tvm/contrib/nvcc.have_tensorcore` 逻辑也提示需要 `-arch=sm_xy`。

---

## 附：关键证据（代码引用）
```930:1119:src/target/source/codegen_cuda.cc
  if (op->op.same_as(builtin::tvm_fill_fragment())) { ... }
  else if (op->op.same_as(builtin::tvm_load_matrix_sync())) { ... }
  else if (op->op.same_as(builtin::tvm_store_matrix_sync())) { ... }
  else if (op->op.same_as(builtin::tvm_mma_sync())) { ... }
  else if (op->op.same_as(builtin::ptx_mma())) { ... PrintMMAAssembly ... }
  else if (op->op.same_as(builtin::ptx_ldmatrix())) { ... PrintLoadMatrixAssembly ... }
```

```520:651:src/target/source/ptx.cc
"mma{.sparse}.sync.aligned{.shape}{.alayout}{.blayout}{.saturate}{.dtype}{.atype}{.btype}{.ctype}{.bitop}"
"ldmatrix.sync.aligned{.shape}{.num}{.trans}{.ss}{.type}"
```

```802:1185:python/tvm/tir/tensor_intrin/cuda.py
WMMA_* TensorIntrin 注册与 `T.tvm_mma_sync`/`T.tvm_load_matrix_sync`/`T.tvm_store_matrix_sync`
```

```201:219:src/tir/transforms/tensorcore_infer_fragment.cc
InferFragment: 收集与校验 fragment 形状/布局并写入 attr
```

---

## 交付物
- 脚本：`mre_wmma_m16n16k16_fp16.py`
- 报告：本文件（Markdown）。
- PTX：运行脚本在具有 CUDA/TVM 的环境生成，搜索 `mma.sync.aligned.m16n16k16` 与 `ldmatrix.sync.aligned` 以佐证。
