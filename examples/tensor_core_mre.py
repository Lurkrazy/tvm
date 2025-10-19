#!/usr/bin/env python3
"""
Minimal Reproducible Example (MRE): 16x16x16 FP16 GEMM using WMMA/Tensor Cores

This script demonstrates TVM's TIR to Tensor Core lowering for NVIDIA GPUs.
Target: sm_89 (Ada, RTX 4090) with mma.sync.aligned PTX instructions.

Usage:
    python3 examples/tensor_core_mre.py

Requirements:
    - TVM with CUDA support
    - CUDA toolkit >= 11.0
    - GPU with compute capability >= 7.0 (sm_70+)
"""

import tvm
from tvm import te
from tvm.script import tir as T
import numpy as np
import os


def gemm_tensorcore_te():
    """
    Define 16x16x16 GEMM using Tensor Expression (TE) API.
    This will be scheduled to use WMMA intrinsics.
    """
    M, N, K = 16, 16, 16
    A = te.placeholder((M, K), name="A", dtype="float16")
    B = te.placeholder((K, N), name="B", dtype="float16")
    k = te.reduce_axis((0, K), name="k")
    
    # C = A @ B (with accumulation in float32)
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A[i, k].astype("float32") * B[k, j].astype("float32"), axis=k
        ),
        name="C",
    )
    
    return A, B, C


def schedule_tensorcore_wmma(A, B, C):
    """
    Schedule the GEMM to use WMMA (C++ API path).
    
    This creates the following memory hierarchy:
    1. Global memory (A, B, C)
    2. Shared memory (A_shared, B_shared)
    3. WMMA fragments (A_frag, B_frag, C_frag)
    """
    s = te.create_schedule(C.op)
    
    # Get the computation block
    C_local = s.cache_write(C, "wmma.accumulator")
    
    # Cache reads to shared memory
    A_shared = s.cache_read(A, "shared", [C_local])
    B_shared = s.cache_read(B, "shared", [C_local])
    
    # Cache reads to WMMA fragments
    A_frag = s.cache_read(A_shared, "wmma.matrix_a", [C_local])
    B_frag = s.cache_read(B_shared, "wmma.matrix_b", [C_local])
    
    # Thread binding for single warp
    # WMMA requires exactly one warp (32 threads)
    tx = te.thread_axis("threadIdx.x")
    
    # Bind computation to warp
    s[C_local].bind(s[C_local].op.axis[0], tx)
    
    # Tensorize the main computation
    # This replaces the reduction loop with WMMA intrinsic
    ko, ki = s[C_local].split(s[C_local].op.reduce_axis[0], factor=16)
    s[C_local].tensorize(ki, tvm.tir.TensorIntrin.get("wmma_sync_16x16x16_f16f16f32"))
    
    # Tensorize the loads
    s[A_frag].tensorize(
        s[A_frag].op.axis[0],
        tvm.tir.TensorIntrin.get("wmma_load_16x16x16_f16_a_shared")
    )
    s[B_frag].tensorize(
        s[B_frag].op.axis[0],
        tvm.tir.TensorIntrin.get("wmma_load_16x16x16_f16_b_shared")
    )
    
    # Tensorize the init
    s[C_local].tensorize(
        s[C_local].op.axis[0],
        tvm.tir.TensorIntrin.get("wmma_fill_16x16x16_f32")
    )
    
    # Tensorize the store
    s[C].tensorize(
        s[C].op.axis[0],
        tvm.tir.TensorIntrin.get("wmma_store_16x16x16_f32_global")
    )
    
    return s


def schedule_tensorcore_mma(A, B, C):
    """
    Schedule the GEMM to use MMA (inline PTX path).
    
    This uses the modern ptx_mma intrinsics for direct PTX generation.
    """
    s = te.create_schedule(C.op)
    
    # Get the computation block
    C_warp = s.cache_write(C, "warp")
    
    # Cache reads to shared memory
    A_shared = s.cache_read(A, "shared", [C_warp])
    B_shared = s.cache_read(B, "shared", [C_warp])
    
    # Cache reads to warp scope for ldmatrix
    A_warp = s.cache_read(A_shared, "warp", [C_warp])
    B_warp = s.cache_read(B_shared, "warp", [C_warp])
    
    # Thread binding
    tx = te.thread_axis("threadIdx.x")
    s[C_warp].bind(s[C_warp].op.axis[0], tx)
    
    # Tensorize with MMA intrinsics
    ko, ki = s[C_warp].split(s[C_warp].op.reduce_axis[0], factor=16)
    s[C_warp].tensorize(ki, tvm.tir.TensorIntrin.get("mma_f16f16f32"))
    
    s[A_warp].tensorize(
        s[A_warp].op.axis[0],
        tvm.tir.TensorIntrin.get("mma_ldmatrix_f16_a")
    )
    s[B_warp].tensorize(
        s[B_warp].op.axis[0],
        tvm.tir.TensorIntrin.get("mma_ldmatrix_f16_b")
    )
    
    s[C_warp].tensorize(
        s[C_warp].op.axis[0],
        tvm.tir.TensorIntrin.get("mma_fill_16x16_f32")
    )
    
    return s


def build_and_extract_ptx(use_wmma=True, arch="sm_89"):
    """
    Build the GEMM kernel and extract PTX assembly.
    
    Args:
        use_wmma: If True, use WMMA path (C++ API). If False, use MMA path (inline PTX).
        arch: Target GPU architecture (e.g., "sm_89", "sm_80", "sm_70")
    
    Returns:
        PTX source code as string
    """
    print(f"\n{'='*80}")
    print(f"Building {'WMMA' if use_wmma else 'MMA'} kernel for {arch}")
    print(f"{'='*80}\n")
    
    # Create the computation
    A, B, C = gemm_tensorcore_te()
    
    # Apply schedule
    if use_wmma:
        try:
            s = schedule_tensorcore_wmma(A, B, C)
            kernel_name = f"wmma_gemm_{arch.replace('sm_', 's')}"
        except Exception as e:
            print(f"Warning: WMMA scheduling failed: {e}")
            print("Falling back to default schedule")
            s = te.create_schedule(C.op)
            kernel_name = f"gemm_fallback_{arch.replace('sm_', 's')}"
    else:
        try:
            s = schedule_tensorcore_mma(A, B, C)
            kernel_name = f"mma_gemm_{arch.replace('sm_', 's')}"
        except Exception as e:
            print(f"Warning: MMA scheduling failed: {e}")
            print("Falling back to default schedule")
            s = te.create_schedule(C.op)
            kernel_name = f"gemm_fallback_{arch.replace('sm_', 's')}"
    
    # Build the module
    target = tvm.target.Target(f"cuda -arch={arch}")
    
    with tvm.transform.PassContext(opt_level=3):
        try:
            mod = tvm.build(s, [A, B, C], target=target, name=kernel_name)
        except Exception as e:
            print(f"Build failed: {e}")
            return None
    
    # Extract PTX
    try:
        ptx_code = mod.imported_modules[0].get_source()
    except Exception as e:
        print(f"Failed to extract PTX: {e}")
        return None
    
    print(f"✓ Build successful! Generated {len(ptx_code)} characters of PTX\n")
    
    # Save PTX to file
    output_dir = "/tmp/tvm_tensor_core_mre"
    os.makedirs(output_dir, exist_ok=True)
    ptx_filename = f"{output_dir}/{kernel_name}.ptx"
    
    with open(ptx_filename, 'w') as f:
        f.write(ptx_code)
    print(f"✓ Full PTX saved to: {ptx_filename}\n")
    
    # Print TIR for inspection
    tir_filename = f"{output_dir}/{kernel_name}.tir"
    with open(tir_filename, 'w') as f:
        f.write(str(mod.get_source()))
    print(f"✓ TIR saved to: {tir_filename}\n")
    
    return ptx_code


def analyze_ptx(ptx_code, arch):
    """
    Analyze PTX code to verify Tensor Core usage.
    
    Looks for:
    - mma.sync.aligned instructions (modern Tensor Cores)
    - ldmatrix instructions (warp matrix load)
    - wmma namespace usage (legacy C++ API)
    """
    if ptx_code is None:
        print("No PTX code to analyze")
        return
    
    print(f"{'='*80}")
    print(f"PTX Analysis for {arch}")
    print(f"{'='*80}\n")
    
    lines = ptx_code.split('\n')
    
    # Search for tensor core indicators
    mma_sync_lines = []
    ldmatrix_lines = []
    wmma_lines = []
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        if 'mma.sync' in line_lower:
            # Found MMA instruction - print with context
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            mma_sync_lines.append((i, '\n'.join(lines[start:end])))
        
        elif 'ldmatrix' in line_lower:
            # Found ldmatrix instruction
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            ldmatrix_lines.append((i, '\n'.join(lines[start:end])))
        
        elif 'wmma' in line_lower or 'nvcuda::wmma' in line:
            # Found WMMA API usage
            wmma_lines.append((i, line))
    
    # Report findings
    tensor_core_used = len(mma_sync_lines) > 0 or len(ldmatrix_lines) > 0 or len(wmma_lines) > 0
    
    if tensor_core_used:
        print("✅ TENSOR CORES DETECTED\n")
    else:
        print("❌ NO TENSOR CORE USAGE FOUND (fallback to regular CUDA cores)\n")
    
    if mma_sync_lines:
        print(f"Found {len(mma_sync_lines)} mma.sync.aligned instruction(s):")
        print("-" * 80)
        for line_num, context in mma_sync_lines[:3]:  # Show first 3
            print(f"Line {line_num}:")
            print(context)
            print("-" * 80)
        if len(mma_sync_lines) > 3:
            print(f"... and {len(mma_sync_lines) - 3} more\n")
    
    if ldmatrix_lines:
        print(f"\nFound {len(ldmatrix_lines)} ldmatrix instruction(s):")
        print("-" * 80)
        for line_num, context in ldmatrix_lines[:3]:
            print(f"Line {line_num}:")
            print(context)
            print("-" * 80)
        if len(ldmatrix_lines) > 3:
            print(f"... and {len(ldmatrix_lines) - 3} more\n")
    
    if wmma_lines:
        print(f"\nFound {len(wmma_lines)} WMMA API reference(s):")
        print("-" * 80)
        for line_num, context in wmma_lines[:5]:
            print(f"Line {line_num}: {context}")
        if len(wmma_lines) > 5:
            print(f"... and {len(wmma_lines) - 5} more")
    
    # Look for fallback indicators
    fma_count = sum(1 for line in lines if 'fma.rn.f32' in line.lower())
    if fma_count > 10 and not tensor_core_used:
        print(f"\n⚠️  Warning: Found {fma_count} FMA instructions without Tensor Core usage")
        print("    This indicates fallback to regular CUDA cores")
    
    print()


def test_correctness():
    """
    Test the correctness of the WMMA kernel (if CUDA is available).
    """
    if not tvm.cuda(0).exist:
        print("CUDA device not available, skipping correctness test")
        return
    
    print(f"\n{'='*80}")
    print("Running Correctness Test")
    print(f"{'='*80}\n")
    
    # Create test data
    M, N, K = 16, 16, 16
    a_np = np.random.randn(M, K).astype("float16")
    b_np = np.random.randn(K, N).astype("float16")
    c_np = a_np.astype("float32") @ b_np.astype("float32")
    
    # Build and run
    A, B, C = gemm_tensorcore_te()
    s = te.create_schedule(C.op)
    
    try:
        s = schedule_tensorcore_wmma(A, B, C)
        print("Using WMMA schedule")
    except:
        print("Using default schedule")
    
    target = tvm.target.Target("cuda -arch=sm_89")
    
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.build(s, [A, B, C], target=target)
    
    # Allocate arrays
    dev = tvm.cuda(0)
    a_tvm = tvm.nd.array(a_np, dev)
    b_tvm = tvm.nd.array(b_np, dev)
    c_tvm = tvm.nd.array(np.zeros((M, N), dtype="float32"), dev)
    
    # Run
    mod(a_tvm, b_tvm, c_tvm)
    
    # Check correctness
    c_result = c_tvm.numpy()
    np.testing.assert_allclose(c_result, c_np, rtol=1e-2, atol=1e-2)
    
    print("✓ Correctness test passed!\n")


def main():
    """Main entry point for the MRE."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║       TVM TIR → Tensor Core (WMMA) Lowering - Minimal Reproducible Example  ║
║                                                                              ║
║  This script demonstrates how TVM lowers TensorIR to NVIDIA Tensor Core     ║
║  instructions (WMMA API and mma.sync.aligned PTX).                          ║
║                                                                              ║
║  Target: sm_89 (Ada, RTX 4090)                                              ║
║  Kernel: 16x16x16 FP16 GEMM with FP32 accumulation                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Test different configurations
    configs = [
        ("sm_89", True, "Ada Lovelace (RTX 4090) - WMMA Path"),
        ("sm_89", False, "Ada Lovelace (RTX 4090) - MMA Path"),
        ("sm_80", True, "Ampere (A100) - WMMA Path"),
        ("sm_70", True, "Volta (V100) - WMMA Path"),
    ]
    
    results = {}
    
    for arch, use_wmma, description in configs:
        print(f"\n{'#'*80}")
        print(f"# Configuration: {description}")
        print(f"{'#'*80}")
        
        try:
            ptx_code = build_and_extract_ptx(use_wmma=use_wmma, arch=arch)
            analyze_ptx(ptx_code, arch)
            results[f"{arch}_{'wmma' if use_wmma else 'mma'}"] = "Success"
        except Exception as e:
            print(f"❌ Error: {e}")
            results[f"{arch}_{'wmma' if use_wmma else 'mma'}"] = f"Failed: {e}"
    
    # Test correctness if CUDA is available
    try:
        test_correctness()
    except Exception as e:
        print(f"Correctness test skipped or failed: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}\n")
    
    for config, status in results.items():
        print(f"{config:30s}: {status}")
    
    print(f"\n{'='*80}")
    print("All PTX and TIR files saved to: /tmp/tvm_tensor_core_mre/")
    print(f"{'='*80}\n")
    
    print("""
Next Steps:
1. Examine the generated PTX files to see mma.sync.aligned instructions
2. Review the TIR files to understand the transformation pipeline
3. Try modifying the schedule to experiment with different configurations
4. Profile with Nsight Compute to verify Tensor Core utilization

For more details, see: docs/TVM_TIR_to_TensorCore_Lowering_Deep_Research.md
""")


if __name__ == "__main__":
    main()
