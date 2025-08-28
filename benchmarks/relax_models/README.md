# TVM Relax Model Benchmarks

This directory contains comprehensive end-to-end benchmarks for TVM Relax auto-tuning capabilities across various deep learning models.

## Overview

The benchmarks demonstrate TVM Relax's auto-tuning performance on:
- **ResNet models** (ResNet18/34/50/101/152)
- **BERT and transformer models** (BERT, RoBERTa, DistilBERT)
- **Modern transformers** (GPT-2, T5, Vision Transformer)
- **Multi-framework comparisons** (PyTorch, ONNX Runtime, TensorRT, OpenVINO)

## Features

### 🚀 Auto-Tuning Capabilities
- **Meta-schedule integration** with evolutionary search strategies
- **Tensor core optimization** for CUDA backends
- **Architecture-specific tuning** for different model types
- **Database caching** for reusable optimization results

### 📊 Performance Analysis
- **Latency measurements** (milliseconds per inference)
- **Throughput analysis** (samples per second)
- **Memory usage tracking**
- **Speedup calculations** vs baseline frameworks

### 🔧 Framework Comparisons
- **PyTorch** native performance
- **ONNX Runtime** with CUDA/CPU providers
- **TensorRT** optimization (NVIDIA GPUs)
- **OpenVINO** acceleration (Intel hardware)

## Quick Start

### Prerequisites

```bash
# Install TVM with Relax support
pip install apache-tvm

# Install deep learning frameworks
pip install torch torchvision transformers

# Optional: Install comparison frameworks
pip install onnxruntime-gpu tensorrt openvino
```

### Running Benchmarks

#### 1. ResNet Benchmark
```bash
# Basic ResNet50 benchmark with auto-tuning
python resnet_benchmark.py --model resnet50 --tune --target cuda

# Compare with other frameworks
python resnet_benchmark.py --model resnet50 --batch-size 8 --compare-frameworks

# Large batch inference
python resnet_benchmark.py --model resnet101 --batch-size 32 --tune
```

#### 2. BERT Benchmark
```bash
# BERT-base with different sequence lengths
python bert_benchmark.py --model bert-base-uncased --seq-length 128 --tune

# BERT-large for production workloads
python bert_benchmark.py --model bert-large-uncased --batch-size 4 --seq-length 512

# Compare with Hugging Face performance
python bert_benchmark.py --model roberta-base --compare-frameworks
```

#### 3. Transformer Benchmark
```bash
# GPT-2 autoregressive generation
python transformer_benchmark.py --model gpt2 --seq-length 1024 --tune

# Vision Transformer for image classification
python transformer_benchmark.py --model google/vit-base-patch16-224 --batch-size 16

# T5 encoder-decoder architecture
python transformer_benchmark.py --model t5-base --seq-length 512 --compare-frameworks
```

#### 4. Multi-Framework Comparison
```bash
# Compare multiple models across all frameworks
python multi_framework_comparison.py --models resnet50,bert-base-uncased,gpt2 --all-frameworks

# Use custom configuration
python multi_framework_comparison.py --config benchmark_config.json
```

## Benchmark Results Structure

Each benchmark produces detailed JSON results:

```json
{
  "model": "resnet50",
  "batch_size": 1,
  "target": "cuda",
  "tuning_enabled": true,
  "tvm_relax": {
    "avg_latency_ms": 12.5,
    "throughput_samples_per_sec": 80.0,
    "total_runs": 100
  },
  "pytorch": {
    "avg_latency_ms": 18.3,
    "throughput_samples_per_sec": 54.6,
    "device": "cuda:0"
  },
  "speedup": 1.46
}
```

## Advanced Usage

### Custom Tuning Configuration

Create a benchmark configuration file:

```json
{
  "models": [
    {
      "name": "resnet50",
      "type": "resnet",
      "batch_size": 8,
      "enable_tuning": true,
      "tuning_trials": 2000
    },
    {
      "name": "bert-large-uncased",
      "type": "bert",
      "batch_size": 4,
      "seq_length": 512,
      "enable_tuning": true
    }
  ],
  "global_settings": {
    "target": "cuda",
    "num_runs": 200,
    "warmup_runs": 20
  }
}
```

### Target-Specific Optimization

```bash
# CUDA with tensor cores
python resnet_benchmark.py --target "cuda -arch=sm_80" --tune

# CPU optimization
python bert_benchmark.py --target "llvm -mcpu=skylake-avx512" --tune

# ARM/mobile targets
python transformer_benchmark.py --target "llvm -mtriple=aarch64-linux-gnu" --tune
```

### Batch Size Optimization

```bash
# Find optimal batch size for throughput
for bs in 1 2 4 8 16 32; do
  python resnet_benchmark.py --model resnet50 --batch-size $bs --tune \
    --output "resnet50_bs${bs}_results.json"
done
```

## Performance Optimization Tips

### 1. Model-Specific Optimizations

**ResNet Models:**
- Enable tensor core usage for FP16 precision
- Use NHWC layout for better memory access patterns
- Optimize convolution fusion patterns

**BERT/Transformer Models:**
- Focus on attention mechanism optimization
- Enable mixed precision training/inference
- Optimize layer normalization and activation functions

**Vision Transformers:**
- Combine CNN and transformer optimizations
- Optimize patch embedding and positional encoding
- Use efficient attention implementations

### 2. Hardware-Specific Tuning

**NVIDIA GPUs:**
- Enable tensor core scheduling rules
- Use CUDA graph optimization for inference
- Optimize memory coalescing patterns

**CPU Inference:**
- Enable SIMD vectorization
- Optimize for cache hierarchy
- Use multi-threading for batch processing

## Interpreting Results

### Key Metrics

1. **Latency (ms)**: Time per inference - lower is better
2. **Throughput (samples/sec)**: Processing rate - higher is better  
3. **Speedup**: Performance improvement vs baseline
4. **Memory Usage**: Peak memory consumption during inference

### Expected Performance Gains

Typical TVM Relax speedups vs other frameworks:

| Model Type | vs PyTorch | vs ONNX RT | vs TensorRT |
|------------|------------|------------|-------------|
| ResNet     | 1.2-2.5x   | 1.1-1.8x   | 0.8-1.2x    |
| BERT       | 1.5-3.0x   | 1.2-2.2x   | 0.9-1.5x    |
| GPT        | 1.3-2.8x   | 1.1-2.0x   | 0.9-1.4x    |
| ViT        | 1.4-2.6x   | 1.2-1.9x   | 0.8-1.3x    |

*Results vary based on model size, batch size, and hardware configuration.*

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or sequence length
2. **Tuning timeout**: Decrease `max_trials_global` parameter
3. **Model conversion errors**: Check model compatibility with TVM Relax
4. **Performance regression**: Verify target configuration and tuning database

### Debug Mode

Enable verbose logging:

```bash
export TVM_LOG_DEBUG=1
python resnet_benchmark.py --model resnet50 --tune
```

### Profiling

Use TVM profiler for detailed analysis:

```python
with tvm.profiling.profiler.profile(
    func, tvm.device(target),
    results_path="profile_results.txt"
):
    vm["main"](input_data)
```

## Contributing

To add new models or frameworks:

1. Create model-specific benchmark class
2. Implement `convert_to_relax()` method
3. Add framework-specific comparison
4. Update multi-framework comparison script
5. Add documentation and examples

## References

- [TVM Relax Documentation](https://tvm.apache.org/docs/tutorial/relax_getting_started.html)
- [Meta-Schedule Auto-Tuning](https://tvm.apache.org/docs/tutorial/meta_schedule_integration.html)
- [Tensor Core Optimization](https://tvm.apache.org/docs/how_to/deploy/tir_tensor_core.html)
- [Performance Tuning Guide](https://tvm.apache.org/docs/tutorial/autotvm_relay_tuning.html)