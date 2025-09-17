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
"""Basic smoke tests for DGDMinimal Search Strategy"""
import tempfile

import tvm
from tvm import meta_schedule as ms
from tvm.script import tir as T
from tvm.target import Target


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (256, 256), "float32")
    B = T.match_buffer(b, (256, 256), "float32")
    C = T.match_buffer(c, (256, 256), "float32")
    for i, j, k in T.grid(256, 256, 256):
        with T.block("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def test_dgd_minimal_cpu():
    # MetaSchedule requires explicit physical core count in target
    from tvm.meta_schedule.utils import cpu_count  # lazy import in test to avoid top-level dep
    ncores = cpu_count(logical=False) or 1
    target = Target(f"llvm -num-cores {ncores}")
    with tempfile.TemporaryDirectory() as work_dir:
        db = ms.tune_tir(
            mod=matmul,
            target=target,
            work_dir=work_dir,
            max_trials_global=64,
            num_trials_per_iter=16,
            strategy=ms.search_strategy.DGDMinimal(
                n_start=2,
                init_size=16,
                slide_window_size=4,
                max_trials=64,
                max_tuning_time_s=30,
                predict_score_ratio=0.5,
                measure_threshold_ratio=0.5,
            ),
        )
        # should have at least one record
        assert len(db.get_all_tuning_records()) > 0


if __name__ == "__main__":
    test_dgd_minimal_cpu()



