/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file per_block_feature.cc
 * \brief Feature extraction for Tensor Core (WMMA) workloads.
 *
 * This feature extractor is designed for Tensor Core schedules and detects
 * WMMA intrinsics (tvm_mma_sync, tvm_load_matrix_sync, etc.) to extract
 * specialized features for cost model prediction.
 *
 * Unlike PerStoreFeature which extracts features per BufferStore,
 * PerBlockFeature extracts features per TIR Block with intrinsic detection.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../tir/transforms/ir_utils.h"
#include "../utils.h"

namespace tvm {
namespace tir {

/*! \brief Given x, compute log2(|x| + 1) */
inline double slog(double x) { return x >= 0 ? std::log2(x + 1) : std::log2(-x + 1); }

// ==================== GPU Architecture Constants ====================
// These constants define GPU hardware characteristics.
// TODO(future): Consider making these configurable per-target.

/*! \brief Number of threads per warp (NVIDIA GPUs) */
constexpr int64_t kWarpSize = 32;

/*! \brief Default feature vector dimension for WMMA intrinsics */
constexpr int64_t kWMMAIntrinFeatureDim = 60;

/*! \brief Maximum number of nested loops to track */
constexpr size_t kMaxTrackedLoops = 8;

/*! \brief Number of behavior types in TCBehavior enum */
constexpr int64_t kNumBehaviorTypes = 5;

// ==================== GPU Architecture Limits ====================
// These are architecture-specific limits that vary by GPU generation.

/*! \brief GPU architecture parameters */
struct GPUArchParams {
  int64_t num_sm = 108;           // Number of SMs (A100: 108, V100: 80, RTX 3090: 82)
  int64_t max_warps_per_sm = 64;  // Max warps per SM (A100: 64, V100: 64)
  int64_t max_blocks_per_sm = 32; // Max blocks per SM
  int64_t max_threads_per_block = 1024;  // Max threads per block

  /*! \brief Get default parameters for common GPUs */
  static GPUArchParams GetDefault() { return GPUArchParams(); }

  /*! \brief Get parameters for specific GPU targets */
  static GPUArchParams FromTarget(const std::string& target_name) {
    GPUArchParams params;

    // ==================== Blackwell (SM 100) ====================
    // NVIDIA RTX 5090 (Blackwell, 2025)
    if (target_name.find("5090") != std::string::npos) {
      params.num_sm = 170;
      params.max_warps_per_sm = 48;
    }

    // ==================== Ada Lovelace (SM 89) ====================
    // NVIDIA RTX 4090
    else if (target_name.find("4090") != std::string::npos) {
      params.num_sm = 128;
      params.max_warps_per_sm = 48;
    }

    // ==================== Ampere (SM 80/86) ====================
    // NVIDIA A100 (SM 80)
    else if (target_name.find("a100") != std::string::npos) {
      params.num_sm = 108;
      params.max_warps_per_sm = 64;
    }
    // NVIDIA RTX 3090 (SM 86)
    else if (target_name.find("3090") != std::string::npos) {
      params.num_sm = 82;
      params.max_warps_per_sm = 48;
    }
    // NVIDIA RTX 3080 Ti (SM 86)
    else if (target_name.find("3080") != std::string::npos) {
      params.num_sm = 80;
      params.max_warps_per_sm = 48;
    }
    // NVIDIA RTX 3050 (SM 86)
    else if (target_name.find("3050") != std::string::npos) {
      params.num_sm = 20;
      params.max_warps_per_sm = 48;
    }

    // ==================== Volta (SM 70) ====================
    // NVIDIA V100
    else if (target_name.find("v100") != std::string::npos) {
      params.num_sm = 80;
      params.max_warps_per_sm = 64;
    }

    // ==================== Turing (SM 75) ====================
    // NVIDIA RTX 2080 Ti
    else if (target_name.find("2080") != std::string::npos) {
      params.num_sm = 68;
      params.max_warps_per_sm = 32;
    }

    // Default: A100-like parameters for unknown targets
    return params;
  }
};

/*!
 * \brief WMMA-specific Group 7 GPU performance features
 *
 * These features capture Tensor Core workload characteristics:
 * - Wave efficiency for WMMA block scheduling
 * - Warp occupancy for concurrent WMMA execution
 * - MMA operation parallelism (ILP equivalent)
 * - Pipeline depth (software pipelining)
 * - Concurrent memory loads
 * - Tile-based data reuse
 * - Operational intensity for global/shared memory
 */
namespace wmma_group7 {

struct Feature {
  /*! \brief Wave efficiency: blocks/SM efficiency */
  double wmma_wave_efficiency = 0.0;
  /*! \brief Warp occupancy for WMMA operations */
  double wmma_warp_occupancy = 0.0;
  /*! \brief Number of mma_sync operations (ILP for WMMA) */
  double wmma_mma_count = 0.0;
  /*! \brief K-loop unroll factor / pipeline depth */
  double wmma_pipeline_depth = 0.0;
  /*! \brief Concurrent memory loads (async copy parallelism) */
  double wmma_concurrent_loads = 0.0;
  /*! \brief Tile reuse factor: (reuse_A + reuse_B) / 2 */
  double wmma_tile_reuse = 0.0;
  /*! \brief Operational intensity for global memory */
  double wmma_oi_global = 0.0;
  /*! \brief Operational intensity for shared memory */
  double wmma_oi_shared = 0.0;

  static constexpr int64_t kCount = 8;

  void Export(std::vector<double>* v) const {
    double vs[] = {
        slog(wmma_wave_efficiency),
        slog(wmma_warp_occupancy),
        slog(wmma_mma_count),
        slog(wmma_pipeline_depth),
        slog(wmma_concurrent_loads),
        slog(wmma_tile_reuse),
        slog(wmma_oi_global),
        slog(wmma_oi_shared),
    };
    v->insert(v->end(), std::begin(vs), std::end(vs));
  }
};

}  // namespace wmma_group7

namespace tensorcore {

/*!
 * \brief Pre-defined feature vectors for WMMA intrinsics.
 * These are extracted from localUniCoMo and represent the characteristics
 * of each WMMA operation.
 */
class WMMAIntrinFeatures {
 public:
  WMMAIntrinFeatures() { InitFeatures(); }

  std::vector<double> GetFeature(const std::string& key) const {
    auto it = intrin_map_.find(key);
    if (it != intrin_map_.end()) {
      return it->second;
    }
    // Return empty feature for unknown intrinsic
    return std::vector<double>(kWMMAIntrinFeatureDim, 0.0);
  }

  bool HasFeature(const std::string& key) const {
    return intrin_map_.find(key) != intrin_map_.end();
  }

 private:
  void InitFeatures() {
    // wmma_sync_16x16x16_f16f16f16: 60-dim feature vector
    // Format: [shape_info(15), compute_intensity(16), buffer_access(29)]
    intrin_map_["wmma_sync_16x16x16_f16f16f16"] = {
        // Shape info (15 dims): Matrix A, B, C dimensions
        0., 16., 0., 0., 0., 1., 16., 0., 0., 0., 2., 16., 0., 0., 0.,
        // Compute intensity (16 dims)
        0., 12.00035218, 12.00035218, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        // Buffer access features (29 dims)
        5.04439412, 4.08746284, 4.08746284, 13.0001761, 9.00281502, 9.00281502, 9.00281502,
        5.04439412, 4.08746284, 4.08746284, 5.04439412, 1., 4.08746284, 13.0001761, 9.00281502,
        9.00281502, 5.04439412, 5.04439412, 4.08746284, 4.08746284, 5.04439412, 1., 4.08746284,
        13.0001761, 9.00281502, 9.00281502, 5.04439412, 1.5849625, 4.08746284};

    // wmma_sync_16x16x16_f16f16f16_trans (transposed version)
    intrin_map_["wmma_sync_16x16x16_f16f16f16_trans"] = {
        0., 16., 0., 0., 0., 1., 16., 0., 0., 0., 2., 16., 0., 0., 0., 0., 12.00035218, 12.00035218,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 5.04439412, 1., 4.08746284, 13.0001761,
        9.00281502, 9.00281502, 5.04439412, 5.04439412, 4.08746284, 4.08746284, 5.04439412, 1.,
        4.08746284, 13.0001761, 9.00281502, 9.00281502, 9.00281502, 5.04439412, 4.08746284,
        4.08746284, 5.04439412, 1., 4.08746284, 13.0001761, 9.00281502, 9.00281502, 5.04439412,
        1.5849625, 4.08746284};

    // wmma_load_16x16x16_f16_a_shared_dyn: 44-dim feature vector
    intrin_map_["wmma_load_16x16x16_f16_a_shared_dyn"] = {
        0., 16., 0., 0., 0., 1., 16., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 5.04439412, 1., 0., 9.00281502, 9.00281502, 9.00281502, 5.04439412,
        4.08746284, 4.08746284, 5.04439412, 1., 0., 9.00281502, 9.00281502, 9.00281502, 5.04439412,
        4.08746284, 4.08746284};

    // wmma_load_16x16x16_f16_b_shared_dyn
    intrin_map_["wmma_load_16x16x16_f16_b_shared_dyn"] = {
        0., 16., 0., 0., 0., 1., 16., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 5.04439412, 1., 0., 9.00281502, 9.00281502, 9.00281502, 5.04439412,
        4.08746284, 4.08746284, 5.04439412, 1., 0., 9.00281502, 9.00281502, 9.00281502, 5.04439412,
        4.08746284, 4.08746284};

    // wmma_fill_16x16x16_f16: 35-dim feature vector
    intrin_map_["wmma_fill_16x16x16_f16"] = {
        0., 16., 0., 0., 0., 1., 16., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 5.04439412, 1., 0., 9.00281502, 9.00281502, 9.00281502, 5.04439412,
        4.08746284, 4.08746284};

    // wmma_store_16x16x16_f16_shared_dyn: 44-dim feature vector
    intrin_map_["wmma_store_16x16x16_f16_shared_dyn"] = {
        0., 16., 0., 0., 0., 1., 16., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 5.04439412, 1., 0., 9.00281502, 9.00281502, 9.00281502, 5.04439412,
        4.08746284, 4.08746284, 5.04439412, 1., 0., 9.00281502, 9.00281502, 9.00281502, 5.04439412,
        4.08746284, 4.08746284};
  }

  std::unordered_map<std::string, std::vector<double>> intrin_map_;
};

// Global instance of WMMA intrinsic features
static WMMAIntrinFeatures g_wmma_features;

/*!
 * \brief Behavior types in a Tensor Core program
 */
enum class BehaviorType : int {
  kPreprocess = 0,  // wmma_fill
  kLoad = 1,        // wmma_load
  kCompute = 2,     // wmma_sync
  kStore = 3,       // wmma_store
  kEpilogue = 4,    // post-processing
};

/*!
 * \brief Data flow source/destination encoding
 * 1 = Global Memory
 * 2 = Shared Memory
 * 3 = Register (Fragment)
 */
struct DataFlow {
  int src = 0;
  int dest = 0;
};

/*!
 * \brief A single behavior in the Tensor Core program
 */
struct TCBehavior {
  BehaviorType type;
  bool is_intrinsic;
  DataFlow data_flow;
  std::string intrinsic_name;
  std::vector<double> features;

  void Export(std::vector<double>* output) const {
    // One-hot encoding for behavior type
    std::vector<double> one_hot(kNumBehaviorTypes, 0.0);
    one_hot[static_cast<int>(type)] = 1.0;
    output->insert(output->end(), one_hot.begin(), one_hot.end());

    // is_intrinsic flag
    output->push_back(is_intrinsic ? 1.0 : 0.0);

    // Data flow (src, dest)
    output->push_back(static_cast<double>(data_flow.src));
    output->push_back(static_cast<double>(data_flow.dest));

    // Intrinsic features
    output->insert(output->end(), features.begin(), features.end());
  }

  static constexpr int64_t kBaseCount = 8;  // one_hot(5) + is_intrin(1) + src(1) + dest(1)
};

/*!
 * \brief GPU loop binding information for WMMA workloads
 */
struct WMMALoopNest {
  int64_t blockIdx_x = 1;
  int64_t blockIdx_y = 1;
  int64_t blockIdx_z = 1;
  int64_t threadIdx_x = 1;
  int64_t threadIdx_y = 1;
  int64_t threadIdx_z = 1;
  int64_t vthread = 1;

  int64_t GridSize() const { return blockIdx_x * blockIdx_y * blockIdx_z; }
  int64_t BlockSize() const { return threadIdx_x * threadIdx_y * threadIdx_z; }
  int64_t NumWarps() const { return (BlockSize() + kWarpSize - 1) / kWarpSize; }
};

/*!
 * \brief WMMA operation statistics for Group 7 features
 */
struct WMMAStats {
  int64_t mma_sync_count = 0;      // Number of mma_sync calls
  int64_t load_a_count = 0;        // Number of load_matrix_sync for A
  int64_t load_b_count = 0;        // Number of load_matrix_sync for B
  int64_t store_count = 0;         // Number of store_matrix_sync
  int64_t fill_count = 0;          // Number of fill_fragment

  // Matrix dimensions (default 16x16x16)
  int wmma_m = 16;
  int wmma_n = 16;
  int wmma_k = 16;

  // Tile dimensions inferred from loops
  int64_t tile_m = 1;
  int64_t tile_n = 1;
  int64_t tile_k = 1;

  // K-loop info for pipeline depth
  int64_t k_loop_extent = 1;
  int64_t k_unroll_factor = 1;

  // Memory info
  int64_t shared_bytes = 0;
  int64_t global_bytes = 0;

  double ComputeFLOPS() const {
    // Each mma_sync: 2 * M * N * K FLOPs
    return 2.0 * wmma_m * wmma_n * wmma_k * mma_sync_count;
  }
};

/*!
 * \brief Represents a Tensor Core program as a sequence of behaviors
 */
struct TCProgram {
  std::vector<TCBehavior> behaviors;
  bool has_tensor_core = false;

  // WMMA Group 7 related data
  WMMALoopNest loop_nest;
  WMMAStats wmma_stats;
  std::unique_ptr<wmma_group7::Feature> group7;

  void ComputeGroup7Features(const GPUArchParams& arch = GPUArchParams::GetDefault()) {
    if (!has_tensor_core) {
      return;
    }

    group7 = std::make_unique<wmma_group7::Feature>();

    // 1. Wave efficiency
    int64_t grid_size = loop_nest.GridSize();
    if (grid_size > 0 && arch.num_sm > 0) {
      double waves = static_cast<double>(grid_size) / arch.num_sm;
      group7->wmma_wave_efficiency = waves / std::ceil(waves);
    }

    // 2. Warp occupancy
    int64_t warps_per_block = loop_nest.NumWarps();
    int64_t blocks_per_sm = std::min(arch.max_blocks_per_sm,
                                     arch.max_warps_per_sm / std::max(warps_per_block, int64_t(1)));
    group7->wmma_warp_occupancy =
        static_cast<double>(warps_per_block * blocks_per_sm) / arch.max_warps_per_sm;

    // 3. MMA count (ILP for WMMA)
    group7->wmma_mma_count = static_cast<double>(wmma_stats.mma_sync_count);

    // 4. Pipeline depth (k-loop unrolling)
    group7->wmma_pipeline_depth = static_cast<double>(wmma_stats.k_unroll_factor);

    // 5. Concurrent loads
    group7->wmma_concurrent_loads = static_cast<double>(wmma_stats.load_a_count + wmma_stats.load_b_count);

    // 6. Tile reuse
    // Reuse is estimated from the ratio of MMA operations to loads
    // reuse_A = mma_sync_count / load_a_count (how many times each A fragment is reused)
    // reuse_B = mma_sync_count / load_b_count (how many times each B fragment is reused)
    double reuse_a = 1.0, reuse_b = 1.0;
    if (wmma_stats.load_a_count > 0) {
      reuse_a = static_cast<double>(wmma_stats.mma_sync_count) / wmma_stats.load_a_count;
    }
    if (wmma_stats.load_b_count > 0) {
      reuse_b = static_cast<double>(wmma_stats.mma_sync_count) / wmma_stats.load_b_count;
    }
    // Use harmonic mean for tile reuse (similar to total_reuse in PerStoreFeature)
    if (reuse_a > 0 && reuse_b > 0) {
      group7->wmma_tile_reuse = 2.0 / (1.0 / reuse_a + 1.0 / reuse_b);
    } else {
      group7->wmma_tile_reuse = (reuse_a + reuse_b) / 2.0;
    }

    // 7. OI Global
    double flops = wmma_stats.ComputeFLOPS();
    if (wmma_stats.global_bytes > 0) {
      group7->wmma_oi_global = flops / wmma_stats.global_bytes;
    }

    // 8. OI Shared
    if (wmma_stats.shared_bytes > 0) {
      group7->wmma_oi_shared = flops / wmma_stats.shared_bytes;
    }
  }

  void Export(std::vector<std::vector<double>>* output) const {
    for (const auto& behavior : behaviors) {
      std::vector<double> feature_vec;
      behavior.Export(&feature_vec);

      // Append Group 7 features if available
      if (group7) {
        group7->Export(&feature_vec);
      }

      output->push_back(feature_vec);
    }
  }

  void AddBehavior(BehaviorType type, bool is_intrinsic, DataFlow flow,
                   const std::string& intrin_name = "") {
    TCBehavior behavior;
    behavior.type = type;
    behavior.is_intrinsic = is_intrinsic;
    behavior.data_flow = flow;
    behavior.intrinsic_name = intrin_name;

    if (is_intrinsic && !intrin_name.empty()) {
      behavior.features = g_wmma_features.GetFeature(intrin_name);
      has_tensor_core = true;
    }

    behaviors.push_back(behavior);
  }
};

/*!
 * \brief Simple loop nest tracking
 */
struct LoopInfo {
  int64_t extent = 1;
  int depth = 0;
};

/*!
 * \brief Collector that visits TIR and detects WMMA intrinsics and BufferStores
 */
class TCFeatureCollector : private StmtExprVisitor {
 public:
  /*!
   * \brief Collect features from an IRModule
   * \param mod The IRModule to analyze
   * \param target Optional target for GPU architecture parameters
   * \return The collected TCProgram with features
   */
  static TCProgram Collect(const IRModule& mod, const ffi::Optional<Target>& target = std::nullopt) {
    TCFeatureCollector collector;
    for (const auto& kv : mod->functions) {
      if (const auto* prim_func = kv.second.as<PrimFuncNode>()) {
        collector.VisitStmt(prim_func->body);
      }
    }
    // Get GPU architecture parameters from target
    GPUArchParams arch = GPUArchParams::GetDefault();
    if (target.defined()) {
      std::string target_str = target.value()->str();
      arch = GPUArchParams::FromTarget(target_str);
    }
    // Compute Group 7 features after collection
    collector.program_.ComputeGroup7Features(arch);
    return std::move(collector.program_);
  }

 private:
  // Track loop nesting and GPU bindings
  void VisitStmt_(const ForNode* loop) final {
    int64_t extent = 1;
    if (const auto* int_imm = loop->extent.as<IntImmNode>()) {
      extent = int_imm->value;
    }

    // Track GPU thread bindings
    std::string thread_tag;
    if (loop->kind == ForKind::kThreadBinding) {
      if (const auto* str_imm = loop->thread_binding.value().as<StringImmNode>()) {
        thread_tag = str_imm->value;
      }
    }

    if (thread_tag == "blockIdx.x") {
      program_.loop_nest.blockIdx_x = extent;
    } else if (thread_tag == "blockIdx.y") {
      program_.loop_nest.blockIdx_y = extent;
    } else if (thread_tag == "blockIdx.z") {
      program_.loop_nest.blockIdx_z = extent;
    } else if (thread_tag == "threadIdx.x") {
      program_.loop_nest.threadIdx_x = extent;
    } else if (thread_tag == "threadIdx.y") {
      program_.loop_nest.threadIdx_y = extent;
    } else if (thread_tag == "threadIdx.z") {
      program_.loop_nest.threadIdx_z = extent;
    } else if (thread_tag == "vthread.x" || thread_tag == "vthread.y" ||
               thread_tag == "vthread.z" || thread_tag == "vthread") {
      program_.loop_nest.vthread *= extent;
    }

    // Track k-loop for pipeline depth estimation
    // K-loops are typically the innermost reduction loops
    bool is_reduction_loop = (loop->kind == ForKind::kSerial && in_wmma_region_);

    loop_depth_++;
    loop_extent_product_ *= extent;
    loop_extents_.push_back(extent);

    // Track potential k-loop
    if (is_reduction_loop && extent > 1) {
      k_loop_candidates_.push_back(extent);
    }

    StmtExprVisitor::VisitStmt_(loop);

    if (is_reduction_loop && extent > 1) {
      k_loop_candidates_.pop_back();
    }

    loop_extents_.pop_back();
    loop_extent_product_ /= extent;
    loop_depth_--;
  }

  // Track allocations for memory info
  void VisitStmt_(const AllocateNode* alloc) final {
    int64_t bytes = 1;
    for (const auto& dim : alloc->extents) {
      if (const auto* imm = dim.as<IntImmNode>()) {
        bytes *= imm->value;
      }
    }
    bytes *= alloc->dtype.bytes();

    ffi::String scope = GetPtrStorageScope(alloc->buffer_var);
    if (scope == "shared" || scope == "shared.dyn") {
      program_.wmma_stats.shared_bytes += bytes;
    } else if (scope == "" || scope == "global") {
      program_.wmma_stats.global_bytes += bytes;
    }

    StmtExprVisitor::VisitStmt_(alloc);
  }

  // Handle regular BufferStore operations
  void VisitStmt_(const BufferStoreNode* store) final {
    StmtExprVisitor::VisitStmt_(store);

    // Check if this store contains WMMA intrinsics
    bool has_wmma_call = false;
    PostOrderVisit(store->value, [&has_wmma_call](const ObjectRef& node) {
      if (const auto* call = node.as<CallNode>()) {
        if (call->op.same_as(builtin::tvm_mma_sync()) ||
            call->op.same_as(builtin::tvm_fill_fragment()) ||
            call->op.same_as(builtin::tvm_load_matrix_sync()) ||
            call->op.same_as(builtin::tvm_store_matrix_sync())) {
          has_wmma_call = true;
        }
      }
    });

    // If no WMMA intrinsics, add as regular compute behavior
    if (!has_wmma_call) {
      AddRegularStoreBehavior(store);
    }
  }

  void AddRegularStoreBehavior(const BufferStoreNode* store) {
    TCBehavior behavior;
    behavior.type = BehaviorType::kCompute;
    behavior.is_intrinsic = false;

    // Determine data flow based on buffer scope
    ffi::String scope = store->buffer.scope();
    int dest = 1;  // default global
    if (scope == "shared" || scope == "shared.dyn") {
      dest = 2;
    } else if (scope == "local" || scope == "wmma.accumulator") {
      dest = 3;
    }
    behavior.data_flow = {1, dest};  // global -> dest

    // Extract loop-based features (same dimension as WMMA intrinsic features)
    std::vector<double> features(kWMMAIntrinFeatureDim, 0.0);

    // Basic loop features
    features[0] = static_cast<double>(loop_depth_);
    features[1] = slog(static_cast<double>(loop_extent_product_));

    // Per-loop extents (up to kMaxTrackedLoops loops)
    for (size_t i = 0; i < std::min(loop_extents_.size(), kMaxTrackedLoops); ++i) {
      features[2 + i] = slog(static_cast<double>(loop_extents_[i]));
    }

    // Arithmetic intensity estimate
    int64_t num_ops = loop_extent_product_;
    features[15] = slog(static_cast<double>(num_ops));  // flop estimate
    features[16] = slog(static_cast<double>(num_ops));  // arithmetic intensity

    // Buffer access features
    int num_loads = 0;
    PostOrderVisit(store->value, [&num_loads](const ObjectRef& node) {
      if (node->IsInstance<BufferLoadNode>()) {
        num_loads++;
      }
    });
    features[30] = slog(static_cast<double>(num_loads));
    features[31] = slog(static_cast<double>(store->buffer->shape.size()));

    // Total touched bytes estimate (use buffer dtype if available, default to float32)
    int64_t bytes_per_elem = store->buffer->dtype.bytes();
    if (bytes_per_elem == 0) bytes_per_elem = 4;  // fallback to float32
    int64_t bytes = loop_extent_product_ * bytes_per_elem;
    features[40] = slog(static_cast<double>(bytes));

    behavior.features = features;
    program_.behaviors.push_back(behavior);
  }

  void VisitExpr_(const CallNode* call) final {
    StmtExprVisitor::VisitExpr_(call);

    // Detect WMMA intrinsics
    if (call->op.same_as(builtin::tvm_fill_fragment())) {
      HandleFillFragment(call);
    } else if (call->op.same_as(builtin::tvm_load_matrix_sync())) {
      HandleLoadMatrixSync(call);
    } else if (call->op.same_as(builtin::tvm_mma_sync())) {
      HandleMmaSync(call);
    } else if (call->op.same_as(builtin::tvm_store_matrix_sync())) {
      HandleStoreMatrixSync(call);
    }
  }

  void HandleFillFragment(const CallNode* call) {
    // wmma_fill: reg -> reg (initialize accumulator)
    // Args: [buffer_var, m, n, k, fill_value]
    int m = 16, n = 16, k = 16;
    if (call->args.size() >= 4) {
      if (const auto* m_imm = call->args[1].as<IntImmNode>()) m = m_imm->value;
      if (const auto* n_imm = call->args[2].as<IntImmNode>()) n = n_imm->value;
      if (const auto* k_imm = call->args[3].as<IntImmNode>()) k = k_imm->value;
    }

    // Update WMMA stats
    program_.wmma_stats.fill_count++;
    in_wmma_region_ = true;

    std::string intrin_name =
        "wmma_fill_" + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k) +
        "_f16";

    DataFlow flow{3, 3};  // reg -> reg
    program_.AddBehavior(BehaviorType::kPreprocess, true, flow, intrin_name);
  }

  void HandleLoadMatrixSync(const CallNode* call) {
    // wmma_load: shared -> reg
    // Args: [buffer_var, m, n, k, index, src_ptr, stride, layout]
    int m = 16, n = 16, k = 16;
    if (call->args.size() >= 4) {
      if (const auto* m_imm = call->args[1].as<IntImmNode>()) m = m_imm->value;
      if (const auto* n_imm = call->args[2].as<IntImmNode>()) n = n_imm->value;
      if (const auto* k_imm = call->args[3].as<IntImmNode>()) k = k_imm->value;
    }

    // Determine matrix type (A or B) based on load count
    bool is_matrix_a = (load_count_ % 2 == 0);
    std::string matrix = is_matrix_a ? "a" : "b";
    load_count_++;

    // Update WMMA stats
    if (is_matrix_a) {
      program_.wmma_stats.load_a_count++;
    } else {
      program_.wmma_stats.load_b_count++;
    }
    in_wmma_region_ = true;

    std::string intrin_name = "wmma_load_" + std::to_string(m) + "x" + std::to_string(n) + "x" +
                              std::to_string(k) + "_f16_" + matrix + "_shared_dyn";

    DataFlow flow{2, 3};  // shared -> reg
    program_.AddBehavior(BehaviorType::kLoad, true, flow, intrin_name);
  }

  void HandleMmaSync(const CallNode* call) {
    // wmma_sync: reg -> reg (compute)
    // Args: [d_buffer, a_buffer, b_buffer, c_buffer, m, n, k, ...]
    int m = 16, n = 16, k = 16;
    // MMA sync typically has shape info later in args
    // For MVP, use default 16x16x16

    // Update WMMA stats
    program_.wmma_stats.mma_sync_count++;
    program_.wmma_stats.wmma_m = m;
    program_.wmma_stats.wmma_n = n;
    program_.wmma_stats.wmma_k = k;
    in_wmma_region_ = true;

    // Estimate k-loop unroll factor from current k-loop candidates
    if (!k_loop_candidates_.empty()) {
      program_.wmma_stats.k_unroll_factor = k_loop_candidates_.back();
    }

    std::string intrin_name = "wmma_sync_" + std::to_string(m) + "x" + std::to_string(n) + "x" +
                              std::to_string(k) + "_f16f16f16";

    DataFlow flow{3, 3};  // reg -> reg
    program_.AddBehavior(BehaviorType::kCompute, true, flow, intrin_name);
  }

  void HandleStoreMatrixSync(const CallNode* call) {
    // wmma_store: reg -> shared
    // Args: [src_buffer, m, n, k, index, dst_ptr, stride, layout]
    int m = 16, n = 16, k = 16;
    if (call->args.size() >= 4) {
      if (const auto* m_imm = call->args[1].as<IntImmNode>()) m = m_imm->value;
      if (const auto* n_imm = call->args[2].as<IntImmNode>()) n = n_imm->value;
      if (const auto* k_imm = call->args[3].as<IntImmNode>()) k = k_imm->value;
    }

    // Update WMMA stats
    program_.wmma_stats.store_count++;

    std::string intrin_name = "wmma_store_" + std::to_string(m) + "x" + std::to_string(n) + "x" +
                              std::to_string(k) + "_f16_shared_dyn";

    DataFlow flow{3, 2};  // reg -> shared
    program_.AddBehavior(BehaviorType::kStore, true, flow, intrin_name);
  }

  TCProgram program_;
  int load_count_ = 0;

  // Loop tracking
  int loop_depth_ = 0;
  int64_t loop_extent_product_ = 1;
  std::vector<int64_t> loop_extents_;

  // WMMA region tracking
  bool in_wmma_region_ = false;
  std::vector<int64_t> k_loop_candidates_;
};

/*!
 * \brief Convert features to NDArray tensor
 */
runtime::Tensor AsTensor(const std::vector<std::vector<double>>& features, int feature_length) {
  int n_rows = features.size();

  runtime::Tensor tensor = runtime::Tensor::Empty(
      /*shape=*/{n_rows, feature_length},
      /*dtype=*/DLDataType{kDLFloat, 64, 1},
      /*ctx=*/DLDevice{kDLCPU, 0});

  if (n_rows == 0) {
    return tensor;
  }

  double* data = static_cast<double*>(tensor->data);

  for (int i = 0; i < n_rows; ++i) {
    const auto& row = features[i];
    for (int j = 0; j < feature_length; ++j) {
      if (j < static_cast<int>(row.size())) {
        data[i * feature_length + j] = row[j];
      } else {
        data[i * feature_length + j] = 0.0;  // Pad with zeros
      }
    }
  }
  return tensor;
}

}  // namespace tensorcore
}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Feature extractor for Tensor Core workloads
 *
 * This extractor detects WMMA intrinsics and extracts specialized features
 * for Tensor Core cost model prediction.
 */
class PerBlockFeatureNode : public FeatureExtractorNode {
 public:
  int feature_vector_length;
  bool extract_workload;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PerBlockFeatureNode>()
        .def_ro("feature_vector_length", &PerBlockFeatureNode::feature_vector_length)
        .def_ro("extract_workload", &PerBlockFeatureNode::extract_workload);
  }

  void ExtractSingle(IRModule mod, const ffi::Optional<Target>& target,
                     std::vector<std::vector<double>>* results) {
    // Collect Tensor Core features with target-specific GPU parameters
    tir::tensorcore::TCProgram program = tir::tensorcore::TCFeatureCollector::Collect(mod, target);

    // Export features
    program.Export(results);

    // If no Tensor Core behaviors found, add a default empty behavior
    if (results->empty()) {
      std::vector<double> empty_feature(feature_vector_length, 0.0);
      results->push_back(empty_feature);
    }
  }

  ffi::Array<runtime::Tensor> ExtractFrom(const TuneContext& tune_context,
                                          const ffi::Array<MeasureCandidate>& candidates) final {
    std::vector<runtime::Tensor> results;
    results.resize(candidates.size());

    // Get target from tune_context for GPU architecture parameters
    ffi::Optional<Target> target = tune_context->target;

    auto f = [this, &candidates, &results, &target](int, int task_id) -> void {
      const auto& candidate = candidates[task_id];
      std::vector<std::vector<double>> features;
      ExtractSingle(DeepCopyIRModule(candidate->sch->mod()), target, &features);
      results[task_id] = tir::tensorcore::AsTensor(features, this->feature_vector_length);
    };

    support::parallel_for_dynamic(0, candidates.size(), tune_context->num_threads, f);
    return results;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("meta_schedule.PerBlockFeature", PerBlockFeatureNode,
                                    FeatureExtractorNode);
};

/*!
 * \brief Create a PerBlockFeature extractor
 */
FeatureExtractor FeatureExtractor::PerBlockFeature(int feature_vector_length,
                                                   bool extract_workload) {
  ObjectPtr<PerBlockFeatureNode> n = ffi::make_object<PerBlockFeatureNode>();
  n->feature_vector_length = feature_vector_length;
  n->extract_workload = extract_workload;
  return FeatureExtractor(n);
}

TVM_FFI_STATIC_INIT_BLOCK() { PerBlockFeatureNode::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("meta_schedule.FeatureExtractorPerBlockFeature",
                        FeatureExtractor::PerBlockFeature);
}

}  // namespace meta_schedule
}  // namespace tvm
