import tvm
import tvm.testing
from tvm import te, auto_scheduler

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

def test_task_scheduler_dynamic_gradient_descent():
    tasks = []

    for i, size in enumerate(sizes):
        M=size[0]
        N=size[1]
        L=size[2]
        print("M=",M,"N=",N,"K=",L, flush=True)
        target = tvm.target.cuda()
        task = tvm.auto_scheduler.SearchTask(func=_matmul, args=(N, L, M, "float32"), target=target)
        
        log_file = "cuda_testCase_" + str(i) +"_matmul_M"+str(M)+"_N"+str(N)+"_K"+str(L)+".json"
        slide_window_size = 3
        max_tuning_time = 99999
        max_trials = 1000
        n_trials = 5
        init_size = 64

        tuner = auto_scheduler.dynamic_gradient_search.DynamicGradientSearchTuner(task, log_file, n_trials, init_size, slide_window_size, max_trials, max_tuning_time)
        tuner.dynamic_gradient_search()
        

if __name__ == "__main__":
    test_task_scheduler_dynamic_gradient_descent()
    
    
    