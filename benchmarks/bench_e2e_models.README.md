# 1) Dependencies
pip install "transformers>=4.41" accelerate sentencepiece
# Please compile and install TVM on your machine (includes Unity/Relax/MetaSchedule)

# 2) Run the default full suite (bs=1/4, seq_len=512, three backends)
python bench_e2e_models.py

# 3) Specify sequence length / tuning trials / run specific backends only
python bench_e2e_models.py --seq-len 128 --tvm-trials 20000 --modes pytorch torch_compile tvm_ms

# 4) Switch to your 4090 target name / or use cuda directly
python bench_e2e_models.py --tvm-target nvidia/geforce-rtx-4090
# Or
python bench_e2e_models.py --tvm-target cuda

