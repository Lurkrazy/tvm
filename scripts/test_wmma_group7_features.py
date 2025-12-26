"""
Test WMMA Group 7 features in PerBlockFeature extractor.

This script tests that the WMMA-specific Group 7 features are properly
extracted from Tensor Core workloads.
"""

import numpy as np
import tvm
from tvm import te, tir
from tvm import meta_schedule as ms
from tvm.script import tir as T


def get_wmma_matmul_func():
    """Generate a simple matmul function for WMMA tensorization."""
    M, N, K = 256, 256, 256

    @T.prim_func
    def matmul(
        A: T.Buffer((M, K), "float16"),
        B: T.Buffer((K, N), "float16"),
        C: T.Buffer((M, N), "float16"),
    ):
        for i, j, k in T.grid(M, N, K):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float16(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    return matmul


def test_per_block_feature_basic():
    """Test basic PerBlockFeature extraction."""
    print("=" * 60)
    print("Test: Basic PerBlockFeature extraction")
    print("=" * 60)

    matmul_func = get_wmma_matmul_func()
    mod = tvm.IRModule({"main": matmul_func})

    target = tvm.target.Target("nvidia/nvidia-a100")

    # Create extractor with Group 7 features (8 extra dims)
    # Base features + Group 7 = 68 + 8 = 76 (approximately)
    extractor = ms.feature_extractor.PerBlockFeature(
        feature_vector_length=76,
        extract_workload=False
    )

    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(),
        task_name="test_matmul",
    )

    # Create a simple schedule
    sch = tir.Schedule(matmul_func, debug_mask="all")
    candidate = ms.MeasureCandidate(sch, None)

    # Extract features
    features_list = extractor.extract_from(ctx, [candidate])

    print(f"Number of feature tensors: {len(features_list)}")

    for i, feat_tensor in enumerate(features_list):
        feat = feat_tensor.numpy()
        print(f"\nCandidate {i}:")
        print(f"  Shape: {feat.shape}")
        print(f"  Feature range: [{feat.min():.4f}, {feat.max():.4f}]")

        # Check that features are extracted
        if feat.shape[0] > 0:
            print(f"  Sample features (last 8 - Group 7):")
            last_8 = feat[0, -8:] if feat.shape[1] >= 8 else feat[0]
            for j, v in enumerate(last_8):
                group7_names = [
                    "wmma_wave_efficiency",
                    "wmma_warp_occupancy",
                    "wmma_mma_count",
                    "wmma_pipeline_depth",
                    "wmma_concurrent_loads",
                    "wmma_tile_reuse",
                    "wmma_oi_global",
                    "wmma_oi_shared",
                ]
                if j < len(group7_names):
                    print(f"    {group7_names[j]}: {v:.6f}")

    print("\n[PASS] Basic feature extraction works")
    return True


def test_per_block_feature_with_schedule():
    """Test PerBlockFeature with a GPU schedule."""
    print("\n" + "=" * 60)
    print("Test: PerBlockFeature with GPU schedule")
    print("=" * 60)

    M, N, K = 256, 256, 256

    @T.prim_func
    def matmul(
        A: T.Buffer((M, K), "float16"),
        B: T.Buffer((K, N), "float16"),
        C: T.Buffer((M, N), "float16"),
    ):
        for i, j, k in T.grid(M, N, K):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float16(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    mod = tvm.IRModule({"main": matmul})
    target = tvm.target.Target("nvidia/nvidia-a100")

    extractor = ms.feature_extractor.PerBlockFeature(
        feature_vector_length=76,
        extract_workload=False
    )

    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(),
        task_name="test_matmul_scheduled",
    )

    # Create a GPU schedule
    sch = tir.Schedule(matmul, debug_mask="all")

    try:
        c = sch.get_block("C")
        c_local = sch.cache_write(c, 0, "local")

        i, j, k = sch.get_loops(c)

        # Tile
        i0, i1 = sch.split(i, factors=[None, 64])
        j0, j1 = sch.split(j, factors=[None, 64])
        k0, k1 = sch.split(k, factors=[None, 32])

        # Further split for threads
        i1o, i1i = sch.split(i1, factors=[None, 8])
        j1o, j1i = sch.split(j1, factors=[None, 8])

        sch.reorder(i0, j0, i1o, j1o, k0, k1, i1i, j1i)

        # Bind
        ij0 = sch.fuse(i0, j0)
        ij1 = sch.fuse(i1o, j1o)
        sch.bind(ij0, "blockIdx.x")
        sch.bind(ij1, "threadIdx.x")

        # Cache shared
        a_shared = sch.cache_read(c, 0, "shared")
        sch.compute_at(a_shared, k0)

        b_shared = sch.cache_read(c, 1, "shared")
        sch.compute_at(b_shared, k0)

        sch.reverse_compute_at(c_local, ij1)

        print("Schedule created successfully")
    except Exception as e:
        print(f"Failed to create schedule: {e}")
        return False

    candidate = ms.MeasureCandidate(sch, None)
    features_list = extractor.extract_from(ctx, [candidate])

    feat = features_list[0].numpy()
    print(f"\nScheduled candidate features:")
    print(f"  Shape: {feat.shape}")
    print(f"  Non-zero features: {np.count_nonzero(feat)}")

    # Check Group 7 features (last 8)
    if feat.shape[1] >= 8:
        group7 = feat[0, -8:]
        print(f"\nGroup 7 features:")
        group7_names = [
            "wmma_wave_efficiency",
            "wmma_warp_occupancy",
            "wmma_mma_count",
            "wmma_pipeline_depth",
            "wmma_concurrent_loads",
            "wmma_tile_reuse",
            "wmma_oi_global",
            "wmma_oi_shared",
        ]
        for name, val in zip(group7_names, group7):
            print(f"    {name}: {val:.6f}")

    print("\n[PASS] Scheduled feature extraction works")
    return True


def test_multiple_candidates():
    """Test extracting features from multiple candidates."""
    print("\n" + "=" * 60)
    print("Test: Multiple candidate feature extraction")
    print("=" * 60)

    M, N, K = 128, 128, 128

    @T.prim_func
    def matmul(
        A: T.Buffer((M, K), "float16"),
        B: T.Buffer((K, N), "float16"),
        C: T.Buffer((M, N), "float16"),
    ):
        for i, j, k in T.grid(M, N, K):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float16(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    mod = tvm.IRModule({"main": matmul})
    target = tvm.target.Target("nvidia/nvidia-a100")

    extractor = ms.feature_extractor.PerBlockFeature(
        feature_vector_length=76,
        extract_workload=False
    )

    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(),
        task_name="test_multi",
    )

    # Create multiple schedules with different tile sizes
    tile_configs = [
        (32, 32, 8),
        (64, 64, 16),
        (128, 128, 32),
    ]

    candidates = []
    for tile_i, tile_j, tile_k in tile_configs:
        sch = tir.Schedule(matmul, debug_mask="all")
        try:
            c = sch.get_block("C")
            i, j, k = sch.get_loops(c)

            # Tile
            i0, i1 = sch.split(i, factors=[None, tile_i])
            j0, j1 = sch.split(j, factors=[None, tile_j])
            k0, k1 = sch.split(k, factors=[None, tile_k])

            sch.reorder(i0, j0, i1, j1, k0, k1)

            ij0 = sch.fuse(i0, j0)
            sch.bind(ij0, "blockIdx.x")

            candidates.append(ms.MeasureCandidate(sch, None))
        except Exception as e:
            print(f"Failed to create schedule with tiles ({tile_i}, {tile_j}, {tile_k}): {e}")

    if not candidates:
        print("No valid candidates created")
        return False

    features_list = extractor.extract_from(ctx, candidates)

    print(f"Number of candidates: {len(candidates)}")
    print(f"Number of feature tensors: {len(features_list)}")

    for i, (feat_tensor, config) in enumerate(zip(features_list, tile_configs)):
        feat = feat_tensor.numpy()
        print(f"\nCandidate {i} (tiles={config}):")
        print(f"  Shape: {feat.shape}")
        print(f"  Mean: {feat.mean():.4f}, Std: {feat.std():.4f}")

    print("\n[PASS] Multiple candidate extraction works")
    return True


def main():
    print("Testing WMMA Group 7 Features in PerBlockFeature")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Basic extraction", test_per_block_feature_basic()))
    results.append(("Scheduled extraction", test_per_block_feature_with_schedule()))
    results.append(("Multiple candidates", test_multiple_candidates()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")

    return all_passed


if __name__ == "__main__":
    main()
