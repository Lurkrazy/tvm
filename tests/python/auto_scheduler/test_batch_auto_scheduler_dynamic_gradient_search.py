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
""" Test task scheduler """

import tempfile

import numpy as np

import tvm
import tvm.testing
from tvm import te, auto_scheduler

from tvm.testing.auto_scheduler import matmul_auto_scheduler_test

sizes=[
    #Bert large
[512,64,1024],      #BMATmul
[512,4096,1024],    #MLP1
    #Bert basic
[512,64,768],       #BMATmul
[512,3072,768],     #MLP1

[512,1024,4096],    #MLP2

[512,768,3072],     #MLP2
]

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def _matmul(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    matmul = te.compute(
        (N, M),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul",
    )
    return [A, B, matmul]

@tvm.testing.requires_llvm
def test_task_scheduler_dynamic_gradient_descent():
    tasks = []
    # for n in [2, 4]:
    #     tasks.append(
    #         auto_scheduler.SearchTask(
    #             func=matmul_auto_scheduler_test, args=(n, n, n), target="cuda"
    #         )
    #     )

    for i, size in enumerate(sizes):
        M=size[0]
        N=size[1]
        L=size[2]
        print("M=",M,"N=",N,"K=",L, flush=True)
        target = tvm.target.cuda()
        # sm_num = 128
        # max_shared_memory_per_block = 48
        # hardware_params = auto_scheduler.HardwareParams(target=target, num_cores=int(sm_num), max_shared_memory_per_block=int(max_shared_memory_per_block)*1024, max_threads_per_block=1024, \
        #                                                 max_vthread_extent=1, vector_unit_bytes=int(999), cache_line_bytes =int(999))

        # task = tvm.auto_scheduler.SearchTask(func=_matmul, args=(N, L, M, "float32"), target=target, hardware_params=hardware_params)
        task = tvm.auto_scheduler.SearchTask(func=_matmul, args=(N, L, M, "float32"), target=target)
        tasks.append(task)

    n_trials = 20
    init_size = 64
    
    for i, task in enumerate(tasks):
        size = sizes[i]
        M = size[0]
        N = size[1]
        L = size[2]
        
        log_file = "cuda_testCase_" + str(i) +"_matmul_M"+str(M)+"_N"+str(N)+"_K"+str(L)+".json"
        slide_window_size = 3
        tuner = auto_scheduler.dynamic_gradient_search.DynamicGradientSearchTuner(task, log_file, n_trials, init_size, slide_window_size)
        tuner.dynamic_gradient_search()
        

if __name__ == "__main__":
    test_task_scheduler_dynamic_gradient_descent()
    
    
    