# 1) 依赖
pip install "transformers>=4.41" accelerate sentencepiece
# TVM 请按你本机编译安装（已含 Unity/Relax/MetaSchedule）

# 2) 跑默认全套（bs=1/4，seq_len=512，三种后端）
python bench_e2e_models.py

# 3) 指定序列长度 / 调优强度 / 只跑某些后端
python bench_e2e_models.py --seq-len 128 --tvm-trials 20000 --modes pytorch torch_compile tvm_ms

# 4) 换成你的 4090 目标名/或直接 cuda
python bench_e2e_models.py --tvm-target nvidia/geforce-rtx-4090
# 或
python bench_e2e_models.py --tvm-target cuda
