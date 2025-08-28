#!/usr/bin/env python3
"""
BERT End-to-End Benchmark with TVM Relax Auto-Tuning

This script demonstrates TVM Relax's auto-tuning capabilities on BERT and other
transformer models. It showcases tensor core utilization for attention mechanisms
and compares performance with Hugging Face Transformers and other frameworks.

Usage:
    python bert_benchmark.py --model bert-base-uncased --seq-length 128 --tune --target cuda
    python bert_benchmark.py --model bert-large-uncased --batch-size 8 --compare-frameworks
"""

import argparse
import time
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

import torch
import torch.nn.functional as F
from torch.fx import symbolic_trace

import tvm
from tvm import relax, meta_schedule as ms
from tvm.relax.frontend.torch import from_fx
from tvm.relax import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BERTBenchmark:
    """BERT benchmark class for TVM Relax auto-tuning evaluation."""
    
    def __init__(self, model_name: str, batch_size: int, seq_length: int,
                 target: str = "cuda", enable_tuning: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.target = tvm.target.Target(target)
        self.enable_tuning = enable_tuning
        
        # Model setup
        self.torch_model, self.tokenizer = self._load_bert_model()
        self.torch_model.eval()
        
        # Input data
        self.input_ids, self.attention_mask = self._prepare_input()
        
        # TVM setup
        self.relax_mod = None
        self.relax_exec = None
        self.tuning_database = None
        
        logger.info(f"Initialized {model_name} benchmark: batch_size={batch_size}, seq_length={seq_length}")
    
    def _load_bert_model(self):
        """Load BERT model from Hugging Face."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        logger.info(f"Loading {self.model_name} from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        
        return model, tokenizer
    
    def _prepare_input(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare sample input for benchmarking."""
        # Create dummy text input
        text = "This is a sample input text for BERT model benchmarking. " * (self.seq_length // 20)
        text = text[:self.seq_length * 5]  # Approximate character limit
        
        # Tokenize
        encoded = self.tokenizer(
            [text] * self.batch_size,
            padding="max_length",
            truncation=True,
            max_length=self.seq_length,
            return_tensors="pt"
        )
        
        return encoded["input_ids"], encoded["attention_mask"]
    
    def convert_to_relax(self) -> tvm.IRModule:
        """Convert BERT model to TVM Relax IR."""
        logger.info("Converting BERT model to TVM Relax IR...")
        
        with torch.no_grad():
            # Create a traced model with specific inputs
            class BERTWrapper(torch.nn.Module):
                def __init__(self, bert_model):
                    super().__init__()
                    self.bert = bert_model
                
                def forward(self, input_ids, attention_mask):
                    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    return outputs.last_hidden_state
            
            wrapped_model = BERTWrapper(self.torch_model)
            
            # Trace the model
            traced_model = symbolic_trace(wrapped_model)
            
            # Define input info for Relax conversion
            input_info = [
                ("input_ids", (self.batch_size, self.seq_length), "int64"),
                ("attention_mask", (self.batch_size, self.seq_length), "int64"),
            ]
            
            # Convert to Relax
            mod = from_fx(traced_model, input_info)
        
        self.relax_mod = mod
        logger.info("BERT model conversion completed")
        return mod
    
    def apply_bert_optimizations(self, mod: tvm.IRModule) -> tvm.IRModule:
        """Apply BERT-specific optimizations."""
        logger.info("Applying BERT-specific optimizations...")
        
        # Transformer-optimized pipeline
        seq = tvm.transform.Sequential([
            # Graph-level optimizations
            relax.transform.DecomposeOpsForInference(),
            relax.transform.CanonicalizeBindings(),
            relax.transform.EliminateCommonSubexpr(),
            
            # Attention-specific optimizations
            relax.transform.CombineParallelMatmul(),  # Crucial for multi-head attention
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
            
            # Memory optimizations for long sequences
            relax.transform.StaticPlanBlockMemory(),
            
            # Layout optimizations for tensor cores (crucial for attention)
            relax.transform.ConvertLayout({
                "relax.nn.dense": ["NT", "auto"],  # Optimize dense layers
                "relax.nn.batch_matmul": ["NTK", "auto"]  # Optimize attention matmul
            }),
            
            relax.transform.FoldConstant(),
            relax.transform.LegalizeOps(),
        ])
        
        optimized_mod = seq(mod)
        logger.info("BERT optimizations completed")
        return optimized_mod
    
    def auto_tune_bert(self, mod: tvm.IRModule, work_dir: str = "./bert_tune_logs") -> tvm.IRModule:
        """Auto-tune BERT model with transformer-specific strategies."""
        if not self.enable_tuning:
            logger.info("Auto-tuning disabled, skipping...")
            return mod
        
        logger.info("Starting BERT auto-tuning with transformer-optimized meta-schedule...")
        
        # Create work directory
        os.makedirs(work_dir, exist_ok=True)
        
        # Configure tuning parameters for transformers
        num_trials = 2000  # Transformers benefit from extensive tuning
        if "base" in self.model_name:
            num_trials = 1500
        elif "large" in self.model_name:
            num_trials = 3000  # Large models need more optimization
        
        # Setup meta-schedule database
        database = ms.tir_integration.database.JSONDatabase(
            path_workload=os.path.join(work_dir, "database_workload.json"),
            path_tuning_record=os.path.join(work_dir, "database_tuning_record.json"),
        )
        
        # Transformer-optimized tuning configuration
        if "cuda" in str(self.target):
            # CUDA-specific optimizations for transformers
            schedule_rules = [
                ms.schedule_rule.MultiLevelTiling(structure="SSSRRSRS"),
                ms.schedule_rule.MultiLevelTilingTensorCore(  # Crucial for attention
                    intrin_groups=[
                        ms.default_config.tensor_core_intrin_groups()["fp16"]
                    ]
                ),
                ms.schedule_rule.AutoInline(
                    into_producer=False,
                    into_consumer=True,
                    inline_const_tensor=True,
                    disallow_if_then_else=True,
                ),
                ms.schedule_rule.CrossThreadReduction(thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]),
                ms.schedule_rule.ParallelizeVectorizeUnroll(
                    max_jobs_per_core=-1,
                    max_vectorize_extent=-1,
                    unroll_max_steps=[0, 16, 64, 512, 1024],
                    unroll_explicit=True,
                ),
            ]
        else:
            schedule_rules = ms.default_config.schedule_rules(target=self.target)
        
        # Auto-tuning pipeline
        with ms.Profiler() as profiler:
            tuned_mod = pipeline.static_shape_tuning_pipeline(
                mod=mod,
                target=self.target,
                work_dir=work_dir,
                max_trials_global=num_trials,
                
                # Advanced tuning for transformers
                num_trials_per_iter=64,  # More trials per iteration
                max_trials_per_task=256,  # More trials per task
                
                # Transformer-optimized schedule rules
                schedule_rules=schedule_rules,
                postprocs=ms.default_config.postproc(target=self.target),
                
                # Cost model tuned for attention patterns
                strategy="evolutionary",
                num_tuning_cores=min(os.cpu_count(), 16),  # Limit for stability
            )
        
        # Save tuning database
        self.tuning_database = database
        
        tuning_time = profiler.get()["total_time"]
        logger.info(f"BERT auto-tuning completed in {tuning_time:.2f} seconds")
        
        return tuned_mod
    
    def build_model(self, mod: tvm.IRModule) -> relax.VirtualMachine:
        """Build the optimized BERT model."""
        logger.info("Building optimized BERT model...")
        
        # Build with optimizations
        exec = relax.build(mod, target=self.target)
        vm = relax.VirtualMachine(exec, tvm.device(str(self.target).split()[0], 0))
        
        self.relax_exec = exec
        logger.info("BERT model build completed")
        return vm
    
    def benchmark_tvm_relax(self, num_runs: int = 50, warmup_runs: int = 5) -> Dict[str, float]:
        """Benchmark TVM Relax BERT performance."""
        if self.relax_exec is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        logger.info(f"Benchmarking TVM Relax BERT ({num_runs} runs)...")
        
        # Create VM and prepare inputs
        vm = relax.VirtualMachine(self.relax_exec, tvm.device(str(self.target).split()[0], 0))
        tvm_input_ids = tvm.nd.array(self.input_ids.numpy())
        tvm_attention_mask = tvm.nd.array(self.attention_mask.numpy())
        
        # Warmup runs
        for _ in range(warmup_runs):
            vm["main"](tvm_input_ids, tvm_attention_mask)
        
        # Synchronize before timing
        tvm.runtime.Device(str(self.target).split()[0], 0).sync()
        
        # Benchmark runs
        start_time = time.time()
        for _ in range(num_runs):
            output = vm["main"](tvm_input_ids, tvm_attention_mask)
        
        # Synchronize and measure
        tvm.runtime.Device(str(self.target).split()[0], 0).sync()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        throughput = self.batch_size / (avg_time / 1000)  # samples/sec
        
        results = {
            "avg_latency_ms": avg_time,
            "throughput_samples_per_sec": throughput,
            "total_runs": num_runs,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length
        }
        
        logger.info(f"TVM Relax BERT Results: {avg_time:.2f}ms avg latency, {throughput:.1f} samples/sec")
        return results
    
    def benchmark_huggingface(self, num_runs: int = 50, warmup_runs: int = 5,
                             use_cuda: bool = True) -> Dict[str, float]:
        """Benchmark Hugging Face Transformers for comparison."""
        logger.info(f"Benchmarking Hugging Face Transformers ({num_runs} runs)...")
        
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        model = self.torch_model.to(device)
        input_ids = self.input_ids.to(device)
        attention_mask = self.attention_mask.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        if use_cuda:
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(input_ids=input_ids, attention_mask=attention_mask)
        
        if use_cuda:
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        throughput = self.batch_size / (avg_time / 1000)  # samples/sec
        
        results = {
            "avg_latency_ms": avg_time,
            "throughput_samples_per_sec": throughput,
            "total_runs": num_runs,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "device": str(device)
        }
        
        logger.info(f"Hugging Face Results: {avg_time:.2f}ms avg latency, {throughput:.1f} samples/sec")
        return results
    
    def run_full_benchmark(self, work_dir: str = "./bert_tune_logs") -> Dict:
        """Run complete BERT benchmark pipeline."""
        logger.info("Starting full BERT benchmark pipeline...")
        
        results = {
            "model": self.model_name,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "target": str(self.target),
            "tuning_enabled": self.enable_tuning,
        }
        
        try:
            # Convert model
            mod = self.convert_to_relax()
            
            # Apply BERT-specific optimizations
            optimized_mod = self.apply_bert_optimizations(mod)
            
            # Auto-tune if enabled
            if self.enable_tuning:
                tuned_mod = self.auto_tune_bert(optimized_mod, work_dir)
            else:
                tuned_mod = optimized_mod
            
            # Build model
            vm = self.build_model(tuned_mod)
            
            # Benchmark TVM Relax
            tvm_results = self.benchmark_tvm_relax()
            results["tvm_relax"] = tvm_results
            
            # Benchmark Hugging Face for comparison
            hf_results = self.benchmark_huggingface()
            results["huggingface"] = hf_results
            
            # Calculate speedup
            speedup = hf_results["avg_latency_ms"] / tvm_results["avg_latency_ms"]
            results["speedup"] = speedup
            
            logger.info(f"BERT benchmark completed. TVM Relax speedup: {speedup:.2f}x")
            
        except Exception as e:
            logger.error(f"BERT benchmark failed: {str(e)}")
            results["error"] = str(e)
            raise
        
        return results


def benchmark_with_onnxruntime(model_name: str, input_ids: np.ndarray, 
                              attention_mask: np.ndarray, num_runs: int = 50) -> Optional[Dict[str, float]]:
    """Benchmark BERT with ONNX Runtime for comparison."""
    try:
        import onnxruntime as ort
        from transformers import AutoModel, AutoTokenizer
        import torch.onnx
        
        logger.info("Benchmarking BERT with ONNX Runtime...")
        
        # Load model
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        # Export to ONNX
        dummy_input_ids = torch.from_numpy(input_ids)
        dummy_attention_mask = torch.from_numpy(attention_mask)
        onnx_path = f"/tmp/{model_name.replace('/', '_')}.onnx"
        
        with torch.no_grad():
            torch.onnx.export(
                model, 
                (dummy_input_ids, dummy_attention_mask),
                onnx_path,
                export_params=True,
                opset_version=11,
                input_names=['input_ids', 'attention_mask'],
                output_names=['last_hidden_state'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
                }
            )
        
        # Load with ONNX Runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Warmup
        for _ in range(5):
            session.run(None, {
                'input_ids': input_ids.astype(np.int64),
                'attention_mask': attention_mask.astype(np.int64)
            })
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            session.run(None, {
                'input_ids': input_ids.astype(np.int64),
                'attention_mask': attention_mask.astype(np.int64)
            })
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000
        throughput = input_ids.shape[0] / (avg_time / 1000)
        
        # Cleanup
        os.remove(onnx_path)
        
        results = {
            "avg_latency_ms": avg_time,
            "throughput_samples_per_sec": throughput,
            "provider": session.get_providers()[0]
        }
        
        logger.info(f"ONNX Runtime BERT Results: {avg_time:.2f}ms avg latency")
        return results
        
    except ImportError:
        logger.warning("ONNX Runtime not available for comparison")
        return None
    except Exception as e:
        logger.error(f"ONNX Runtime BERT benchmark failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="BERT TVM Relax Auto-Tuning Benchmark")
    parser.add_argument("--model", default="bert-base-uncased", 
                       help="BERT model name from Hugging Face")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--target", default="cuda", help="TVM target (cuda, cpu, etc.)")
    parser.add_argument("--tune", action="store_true", help="Enable auto-tuning")
    parser.add_argument("--compare-frameworks", action="store_true", 
                       help="Compare with other frameworks")
    parser.add_argument("--work-dir", default="./bert_tune_logs", 
                       help="Directory for tuning logs")
    parser.add_argument("--output", default="bert_benchmark_results.json", 
                       help="Output file for results")
    parser.add_argument("--num-runs", type=int, default=50, 
                       help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = BERTBenchmark(
        model_name=args.model,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        target=args.target,
        enable_tuning=args.tune
    )
    
    # Run benchmark
    results = benchmark.run_full_benchmark(args.work_dir)
    
    # Add additional framework comparisons
    if args.compare_frameworks:
        # ONNX Runtime comparison
        onnx_results = benchmark_with_onnxruntime(
            args.model, 
            benchmark.input_ids.numpy(), 
            benchmark.attention_mask.numpy(),
            args.num_runs
        )
        if onnx_results:
            results["onnx_runtime"] = onnx_results
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("BERT BENCHMARK SUMMARY")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sequence Length: {args.seq_length}")
    print(f"Target: {args.target}")
    print(f"Auto-tuning: {'Enabled' if args.tune else 'Disabled'}")
    print()
    
    if "tvm_relax" in results:
        tvm_lat = results["tvm_relax"]["avg_latency_ms"]
        tvm_thr = results["tvm_relax"]["throughput_samples_per_sec"]
        print(f"TVM Relax:       {tvm_lat:.2f}ms  ({tvm_thr:.1f} samples/sec)")
    
    if "huggingface" in results:
        hf_lat = results["huggingface"]["avg_latency_ms"]
        hf_thr = results["huggingface"]["throughput_samples_per_sec"]
        print(f"Hugging Face:    {hf_lat:.2f}ms  ({hf_thr:.1f} samples/sec)")
    
    if "onnx_runtime" in results:
        ort_lat = results["onnx_runtime"]["avg_latency_ms"]
        ort_thr = results["onnx_runtime"]["throughput_samples_per_sec"]
        print(f"ONNX Runtime:    {ort_lat:.2f}ms  ({ort_thr:.1f} samples/sec)")
    
    if "speedup" in results:
        print(f"\nTVM Relax Speedup: {results['speedup']:.2f}x vs Hugging Face")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()