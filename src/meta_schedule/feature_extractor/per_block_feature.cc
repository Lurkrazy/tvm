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
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace tir {

/*! \brief Given x, compute log2(|x| + 1) */
inline double slog(double x) { return x >= 0 ? std::log2(x + 1) : std::log2(-x + 1); }

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
    return std::vector<double>(60, 0.0);
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
    // One-hot encoding for behavior type (5 dims)
    std::vector<double> one_hot(5, 0.0);
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
 * \brief Represents a Tensor Core program as a sequence of behaviors
 */
struct TCProgram {
  std::vector<TCBehavior> behaviors;
  bool has_tensor_core = false;

  void Export(std::vector<std::vector<double>>* output) const {
    for (const auto& behavior : behaviors) {
      std::vector<double> feature_vec;
      behavior.Export(&feature_vec);
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
  static TCProgram Collect(const IRModule& mod) {
    TCFeatureCollector collector;
    for (const auto& kv : mod->functions) {
      if (const auto* prim_func = kv.second.as<PrimFuncNode>()) {
        collector.VisitStmt(prim_func->body);
      }
    }
    return collector.program_;
  }

 private:
  // Track loop nesting
  void VisitStmt_(const ForNode* loop) final {
    int64_t extent = 1;
    if (const auto* int_imm = loop->extent.as<IntImmNode>()) {
      extent = int_imm->value;
    }

    loop_depth_++;
    loop_extent_product_ *= extent;
    loop_extents_.push_back(extent);

    StmtExprVisitor::VisitStmt_(loop);

    loop_extents_.pop_back();
    loop_extent_product_ /= extent;
    loop_depth_--;
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

    // Extract loop-based features (60 dims to match WMMA intrinsic features)
    std::vector<double> features(60, 0.0);

    // Basic loop features
    features[0] = static_cast<double>(loop_depth_);
    features[1] = slog(static_cast<double>(loop_extent_product_));

    // Per-loop extents (up to 8 loops)
    for (size_t i = 0; i < std::min(loop_extents_.size(), size_t(8)); ++i) {
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

    // Total touched bytes estimate
    int64_t bytes = loop_extent_product_ * 4;  // assume float32
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
    std::string matrix = (load_count_ % 2 == 0) ? "a" : "b";
    load_count_++;

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

  void ExtractSingle(IRModule mod, std::vector<std::vector<double>>* results) {
    // Collect Tensor Core features
    tir::tensorcore::TCProgram program = tir::tensorcore::TCFeatureCollector::Collect(mod);

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

    auto f = [this, &candidates, &results](int, int task_id) -> void {
      const auto& candidate = candidates[task_id];
      std::vector<std::vector<double>> features;
      ExtractSingle(DeepCopyIRModule(candidate->sch->mod()), &features);
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
