# Minimal Repro: 16x16x16 WMMA FP16 GEMM producing mma.sync.aligned on sm_89
# Requires TVM built with CUDA. This script constructs TensorIR + tensorize to WMMA
# and prints PTX to confirm mma.sync.aligned.m16n16k16.

import tvm
from tvm import te, tir
from tvm.script import tir as T
from tvm.tir.tensor_intrin.cuda import (
    LDMATRIX_f16_A_INTRIN,
    LDMATRIX_f16_B_TRANS_INTRIN,
    WMMA_FILL_16x16x16_F32_INTRIN,
    WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN,
    WMMA_SYNC_16x16x16_f16f16f32_TRANS_INTRIN,
)

# 1) Define naive matmul 16x16x16 (A row-major, B col-major for WMMA)
@tvm.script.ir_module
class Matmul16x16x16:
    @T.prim_func
    def main(A: T.Buffer((16, 16), "float16"),
             B: T.Buffer((16, 16), "float16"),
             C: T.Buffer((16, 16), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(16, 16, 16):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + T.cast(A[vi, vk], "float32") * T.cast(B[vk, vj], "float32")

mod = Matmul16x16x16
sch = tir.Schedule(mod)
block = sch.get_block("C")
# Tile to 16x16x16 and create warp/shared caches
i, j, k = sch.get_loops(block)
i0, i1 = sch.split(i, factors=[None, 16])
j0, j1 = sch.split(j, factors=[None, 16])
k0, k1 = sch.split(k, factors=[None, 16])
sch.reorder(i0, j0, k0, i1, j1, k1)
block_inner = sch.blockize(i1)
block_outer, block_inner = block_inner, block

# cache into shared and warp scopes
A_shared = sch.cache_read(block_outer, 0, "shared")
B_shared = sch.cache_read(block_outer, 1, "shared")
A_warp = sch.cache_read(block_outer, 0, "wmma.matrix_a")
B_warp = sch.cache_read(block_outer, 1, "wmma.matrix_b")
C_warp = sch.cache_write(block_outer, 0, "wmma.accumulator")

# Move reads/writes
_, _, k0 = sch.get_loops(block_outer)
sch.compute_at(A_shared, k0)
sch.compute_at(B_shared, k0)
sch.compute_at(A_warp, k0)
sch.compute_at(B_warp, k0)
sch.reverse_compute_at(C_warp, sch.get_loops(block_outer)[1])

# Tensorize with WMMA intrinsics (ldmatrix + wmma.sync + store)
# Use B transposed variant (WMMA expects B col-major via *_TRANS intrin)
# Load fragments from shared into wmma.matrix_* via ldmatrix wrappers
loop_a = sch.get_loops(A_warp)[-2]
loop_b = sch.get_loops(B_warp)[-2]
sch.tensorize(loop_a, LDMATRIX_f16_A_INTRIN)
sch.tensorize(loop_b, LDMATRIX_f16_B_TRANS_INTRIN)

# Compute and init/store tensorization
ki = sch.get_loops(block_inner)[-1]
init = sch.decompose_reduction(block_outer, k0)
init_loop = sch.get_loops(init)[-2]
store_loop = sch.get_loops(C_warp)[-2]

sch.tensorize(ki, WMMA_SYNC_16x16x16_f16f16f32_TRANS_INTRIN)
sch.tensorize(init_loop, WMMA_FILL_16x16x16_F32_INTRIN)
sch.tensorize(store_loop, WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN)

rt_mod = tvm.build(sch.mod, target="cuda -arch=sm_89")

# Try to extract PTX
ptx = None
try:
    ptx = rt_mod.get_source("ptx")
except Exception:
    try:
        ptx = rt_mod.imported_modules[0].get_source()
    except Exception:
        pass

print("==== Generated PTX (truncated) ====")
if ptx:
    # Print only lines with mma.sync.aligned for evidence
    lines = [l for l in ptx.splitlines() if "mma.sync.aligned" in l]
    print("\n".join(lines[:10]))
else:
    print("<No PTX available; ensure TVM with CUDA is installed>")
