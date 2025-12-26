# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Feature extractor for Tensor Core (WMMA) workloads.

This feature extractor detects WMMA intrinsics (tvm_mma_sync, tvm_load_matrix_sync, etc.)
and extracts specialized features for Tensor Core cost model prediction.

Unlike PerStoreFeature which extracts features per BufferStore,
PerBlockFeature extracts features per TIR Block with intrinsic detection.
"""
from tvm_ffi import register_object

from .. import _ffi_api
from .feature_extractor import FeatureExtractor


@register_object("meta_schedule.PerBlockFeature")
class PerBlockFeature(FeatureExtractor):
    """PerBlockFeature extracts features for Tensor Core workloads.

    This feature extractor is designed for Tensor Core schedules and detects
    WMMA intrinsics to extract specialized features for cost model prediction.

    Parameters
    ----------
    feature_vector_length : int
        Length of the output feature vector.
        Default is 68 (8 base features + 60 WMMA intrinsic features).
    extract_workload : bool
        Whether to extract features in the workload in tuning context or not.

    Examples
    --------
    .. code-block:: python

        # For Tensor Core tuning
        feature_extractor = PerBlockFeature()

        # Or use the factory method
        feature_extractor = FeatureExtractor.create("per-block-feature")
    """

    feature_vector_length: int
    """Length of the feature vector."""
    extract_workload: bool
    """Whether to extract features in the workload in tuning context or not."""

    def __init__(
        self,
        feature_vector_length: int = 68,
        extract_workload: bool = False,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.FeatureExtractorPerBlockFeature,  # type: ignore # pylint: disable=no-member
            feature_vector_length,
            extract_workload,
        )
