import tvm
import tvm.testing
from tvm import te, auto_scheduler, topi
import json
import os
import csv
import time
import threading

########### all pz 
sizesResnet = [
    [1, 224, 224, 64, 3, 7, 7, 2, 3],   # RESNET1
    [1, 56, 56, 64, 64, 1, 1, 1, 0],    # RESNET2
    [1, 56, 56, 64, 64, 3, 3, 1, 1],    # RESNET2
    [1, 56, 56, 256, 64, 1, 1, 1, 0],   # RESNET2
    [1, 56, 56, 128, 256, 1, 1, 2, 0],  # RESNET3
    [1, 28, 28, 128, 128, 3, 3, 1, 1],  # RESNET3
    [1, 28, 28, 512, 128, 1, 1, 1, 0],  # RESNET3
    [1, 28, 28, 256, 512, 1, 1, 2, 0],  # RESNET4
    [1, 14, 14, 256, 256, 3, 3, 1, 1],  # RESNET4
    [1, 14, 14, 1024, 256, 1, 1, 1, 0], # RESNET4
    [1, 14, 14, 512, 1024, 1, 1, 2, 0], # RESNET5
    [1, 7, 7, 512, 512, 3, 3, 1, 1],    # RESNET5
    [1, 7, 7, 2048, 512, 1, 1, 1, 0],   # RESNET5
]

sizesYolo = [
    [1, 544, 544, 32, 3, 3, 3, 1, 1],    # Yolo0
    [1, 272, 272, 64, 32, 3, 3, 1, 1],   # Yolo2
    [1, 136, 136, 128, 64, 3, 3, 1, 1],  # yolo4
    [1, 136, 136, 64, 128, 1, 1, 1, 0],  # yolo5
    [1, 68, 68, 256, 128, 3, 3, 1, 1],   # yolo8
    [1, 68, 68, 128, 256, 1, 1, 1, 0],   # yolo9
    [1, 68, 68, 512, 256, 3, 3, 1, 1],   # yolo14
    [1, 34, 34, 512, 256, 3, 3, 1, 1],   # yolo12
    [1, 34, 34, 256, 512, 1, 1, 1, 0],   # yolo13
    [1, 17, 17, 1024, 512, 3, 3, 1, 1],  # yolo18
    [1, 17, 17, 512, 1024, 1, 1, 1, 0],  # yolo19
]
########### all pz end

class Conv2DParams:
    def __init__(self, N, H, W, CO, CI, KH, KW, strides, padding):
        self.N = N
        self.H = H
        self.W = W
        self.CO = CO
        self.CI = CI
        self.KH = KH
        self.KW = KW
        self.strides = strides
        self.padding = padding
        
@auto_scheduler.register_workload
def conv2d(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]


def parse_and_write(log_file_path, csv_file_path, start_time):
    if not os.path.exists(log_file_path):
        log_file_path = "x" + log_file_path
    # deal with FileNotFoundError exception
    try:
        with open(log_file_path, 'r') as log_file:
            log_content = log_file.read()
    except FileNotFoundError:
        # print(f'Json log might not be generated yet. Skip this round.', flush=True)
        return

    lines = log_content.splitlines()
    num_lines = len(lines)
    min_time_value = float('inf')

    for line in lines:
        try:
            data = json.loads(line)
            time_value = data['r'][0][0]

            if time_value < min_time_value:
                min_time_value = time_value
        except json.JSONDecodeError:
            continue

    if min_time_value != float('inf'):
        elapsed_time = int(time.time() - start_time)
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([elapsed_time, min_time_value, num_lines])

timer = None

def start_timer(interval, log_file_path, csv_file_path, start_time):
    global timer
    timer = threading.Timer(interval, start_timer, [interval, log_file_path, csv_file_path, start_time])
    timer.start()
    parse_and_write(log_file_path, csv_file_path, start_time)
    
def stop_timer():
    global timer
    if timer:
        timer.cancel()

def test_task_scheduler_dynamic_gradient_descent(network = "resnet"):
    
    if network == "yolo":
        sizes=sizesYolo
        print(f"\ntesting yolo with {len(sizes)} layers\n")
    elif network == "resnet":
        sizes=sizesResnet
        print(f"\ntesting resnet with {len(sizes)} layers\n")
    else:
        raise Exception("network not specified!")
        
    conv_params = {}
    for i, size in enumerate(sizes):
        N, H, W, CO, CI, KH, KW, stride, pad = size
        key = "conv" + str(i+1)
        #N, H, W, CO, CI, KH, KW, strides, padding
        conv_params[key] = Conv2DParams(N, H, W, CO, CI, KH, KW, (stride, stride), (pad, pad))
        
    for i, key in enumerate(conv_params.keys()):
        conv = conv_params[key]
        target = tvm.target.cuda()
        
        # Use the conv2d layer to test
        N, H, W, CO, CI, KH, KW, strides, padding = conv.N, conv.H, conv.W, conv.CO, conv.CI, conv.KH, conv.KW, conv.strides, conv.padding
                
        task = auto_scheduler.SearchTask(
            func=conv2d, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target,
        )
        
        log_file = "cuda_"+network+"_testCase_"+str(i)+"_conv2d_N_"+str(N)+"_H_"+str(H)+"_W_"+str(W)+"_CO_"+str(CO)+"_CI_"+str(CI)+"_KH_"+str(KH)+"_KW_"+str(KW)+"_strides_"+str(strides)+"_padding_"+str(padding)+".json"
        csv_file_path = log_file.replace('.json', '.csv')
        
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["elapsed_time", "min_time_value", "num_configs"])
        interval = 1  # 1 second
        start_time = time.time()

        start_timer(interval, log_file, csv_file_path, start_time)

        slide_window_size = 3
        max_tuning_time = 99999
        max_trials = 1000
        n_trials = 5
        init_size = 64
        
        tuner = auto_scheduler.dynamic_gradient_search.DynamicGradientSearchTuner(task, log_file, n_trials, init_size, slide_window_size, max_trials, max_tuning_time)
        tuner.dynamic_gradient_search()
        
        end_time = time.time()
        search_time = end_time - start_time
        search_time /= 60
        time.sleep(3)
        stop_timer()
        print(f"Total search time: {search_time} minutes", flush=True)

if __name__ == "__main__":
    
    import sys
    if len(sys.argv) > 1:
        network = sys.argv[1]
        if network == "yolo":
            test_task_scheduler_dynamic_gradient_descent("yolo")
        elif network == "resnet":
            test_task_scheduler_dynamic_gradient_descent("resnet")
    else:
        test_task_scheduler_dynamic_gradient_descent()
    
    
    