"""
Analyze correlation between GPU features and schedule quality.

This script generates schedules with different tile sizes and analyzes
the correlation between features and expected performance characteristics.
"""

import numpy as np
import tvm
from tvm import te, tir
from tvm import meta_schedule as ms
from tvm.script import tir as T
from scipy import stats
import itertools

@T.prim_func
def matmul(
    A: T.Buffer((1024, 1024), "float32"),
    B: T.Buffer((1024, 1024), "float32"),
    C: T.Buffer((1024, 1024), "float32"),
):
    for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def create_gpu_schedule(tile_i, tile_j, tile_k, thread_i, thread_j):
    """Create a GPU schedule with specified tile sizes."""
    func = matmul
    sch = tir.Schedule(func, debug_mask="all")

    try:
        c = sch.get_block("C")
        c_local = sch.cache_write(c, 0, "local")
        i, j, k = sch.get_loops(c)

        # Tile the loops
        i0, i1 = sch.split(i, factors=[None, tile_i])
        j0, j1 = sch.split(j, factors=[None, tile_j])
        k0, k1 = sch.split(k, factors=[None, tile_k])

        # Further split for threads
        i1o, i1i = sch.split(i1, factors=[None, thread_i])
        j1o, j1i = sch.split(j1, factors=[None, thread_j])

        sch.reorder(i0, j0, i1o, j1o, k0, k1, i1i, j1i)

        # Fuse and bind
        ij0 = sch.fuse(i0, j0)
        ij1 = sch.fuse(i1o, j1o)
        sch.bind(ij0, "blockIdx.x")
        sch.bind(ij1, "threadIdx.x")

        # Cache to shared memory
        a_shared = sch.cache_read(c, 0, "shared")
        sch.compute_at(a_shared, k0)

        b_shared = sch.cache_read(c, 1, "shared")
        sch.compute_at(b_shared, k0)

        # Write back
        sch.reverse_compute_at(c_local, ij1)

        return sch
    except Exception as e:
        return None


def estimate_performance(tile_i, tile_j, tile_k, thread_i, thread_j):
    """Estimate relative performance score based on tile sizes.

    Better schedules typically have:
    - Larger tiles (better data reuse)
    - Balanced thread dimensions (better occupancy)
    - tile_k that allows good shared memory usage
    """
    M, N, K = 1024, 1024, 1024

    # Compute various performance proxies
    num_blocks = (M // tile_i) * (N // tile_j)
    threads_per_block = (tile_i // thread_i) * (tile_j // thread_j)

    # Wave efficiency (prefer full waves)
    num_sm = 108  # A100
    waves = num_blocks / num_sm
    wave_eff = waves / np.ceil(waves) if waves > 0 else 0

    # Occupancy (prefer balanced threads)
    max_threads = 1024
    occ = min(threads_per_block / max_threads, 1.0)

    # Data reuse (larger tiles = better reuse)
    reuse_score = (tile_i * tile_j * tile_k) / (K * 64)  # normalize

    # Shared memory efficiency
    shared_per_block = (tile_i * tile_k + tile_k * tile_j) * 4  # bytes
    max_shared = 48 * 1024  # 48KB
    shared_eff = 1.0 if shared_per_block < max_shared else max_shared / shared_per_block

    # Combined score (higher is better)
    score = wave_eff * 0.2 + occ * 0.3 + min(reuse_score, 1.0) * 0.3 + shared_eff * 0.2

    return score


def main():
    target = tvm.target.Target("nvidia/nvidia-a100")
    mod = tvm.IRModule({"main": matmul})

    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(),
        task_name="matmul",
    )

    extractor = ms.feature_extractor.PerStoreFeature()

    # Generate schedules with different configurations
    tile_options = [
        # (tile_i, tile_j, tile_k, thread_i, thread_j)
        (32, 32, 8, 4, 4),
        (32, 32, 16, 4, 4),
        (32, 32, 32, 4, 4),
        (64, 64, 8, 8, 8),
        (64, 64, 16, 8, 8),
        (64, 64, 32, 8, 8),
        (128, 128, 8, 16, 8),
        (128, 128, 16, 16, 8),
        (128, 128, 32, 16, 8),
        (64, 32, 16, 8, 4),
        (32, 64, 16, 4, 8),
        (128, 64, 16, 16, 8),
        (64, 128, 16, 8, 16),
        (256, 128, 8, 16, 16),
        (128, 256, 8, 16, 16),
        (32, 32, 4, 4, 4),
        (16, 16, 8, 4, 4),
        (64, 64, 4, 8, 8),
        (32, 128, 16, 4, 16),
        (128, 32, 16, 16, 4),
    ]

    all_features = []
    all_scores = []

    print("Generating schedules and extracting features...")
    for config in tile_options:
        tile_i, tile_j, tile_k, thread_i, thread_j = config

        sch = create_gpu_schedule(tile_i, tile_j, tile_k, thread_i, thread_j)
        if sch is None:
            continue

        try:
            # Extract features
            features_list = extractor.extract_from(ctx, [ms.MeasureCandidate(sch, None)])
            features = features_list[0].numpy()

            # Average across stores
            avg_features = np.mean(features, axis=0)

            # Estimate performance score
            score = estimate_performance(tile_i, tile_j, tile_k, thread_i, thread_j)

            all_features.append(avg_features)
            all_scores.append(score)

            print(f"  Config {config}: score={score:.4f}")
        except Exception as e:
            print(f"  Config {config}: failed - {e}")
            continue

    print(f"\nTotal valid schedules: {len(all_features)}")

    if len(all_features) < 5:
        print("Not enough samples for analysis")
        return

    features_matrix = np.array(all_features)
    scores = np.array(all_scores)

    # Feature names for Group 7
    group7_names = [
        "wave_efficiency",
        "est_occupancy",
        "ilp",
        "wlp",
        "mlp",
        "total_reuse",
        "oi_global",
        "oi_shared",
    ]

    # Calculate correlations
    print("\n" + "=" * 80)
    print("Group 7 GPU Features Correlation with Performance Score")
    print("=" * 80)
    print(f"{'Feature':<20} {'Pearson r':>12} {'p-value':>12} {'Spearman ρ':>12} {'p-value':>12}")
    print("-" * 80)

    correlations = []
    for i, name in enumerate(group7_names):
        feat_idx = -(8 - i)
        feat_values = features_matrix[:, feat_idx]

        if np.std(feat_values) < 1e-10:
            print(f"{name:<20} {'N/A (no variance)':>50}")
            correlations.append((name, 0, 1, 0, 1))
            continue

        pearson_r, pearson_p = stats.pearsonr(feat_values, scores)
        spearman_r, spearman_p = stats.spearmanr(feat_values, scores)

        correlations.append((name, pearson_r, pearson_p, spearman_r, spearman_p))

        sig = ""
        if spearman_p < 0.001:
            sig = "***"
        elif spearman_p < 0.01:
            sig = "**"
        elif spearman_p < 0.05:
            sig = "*"

        print(f"{name:<20} {pearson_r:>12.4f} {pearson_p:>12.4e} {spearman_r:>12.4f} {spearman_p:>12.4e} {sig}")

    print("-" * 80)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")

    # Summary
    print("\n" + "=" * 80)
    print("Summary: Features with |Spearman ρ| > 0.3")
    print("=" * 80)
    significant_features = [(name, sr) for name, pr, pp, sr, sp in correlations if abs(sr) > 0.3]
    if significant_features:
        for name, sr in sorted(significant_features, key=lambda x: abs(x[1]), reverse=True):
            direction = "↑ higher is better" if sr > 0 else "↓ lower is better"
            print(f"  {name}: ρ={sr:.4f} ({direction})")
    else:
        print("  No features with |ρ| > 0.3 found")

    # Print feature statistics
    print("\n" + "=" * 80)
    print("Feature Value Statistics")
    print("=" * 80)
    print(f"{'Feature':<20} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("-" * 68)
    for i, name in enumerate(group7_names):
        feat_idx = -(8 - i)
        feat_values = features_matrix[:, feat_idx]
        print(f"{name:<20} {feat_values.min():>12.4f} {feat_values.max():>12.4f} {feat_values.mean():>12.4f} {feat_values.std():>12.4f}")


if __name__ == "__main__":
    main()
