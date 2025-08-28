#!/usr/bin/env python3
"""
Transformer Models End-to-End Benchmark with TVM Relax Auto-Tuning

This script demonstrates TVM Relax's auto-tuning capabilities on various transformer
architectures including GPT, ViT, and T5. It showcases advanced optimizations for
attention mechanisms, layer normalization, and feed-forward networks.

Usage:
    python transformer_benchmark.py --model gpt2 --seq-length 512 --tune --target cuda
    python transformer_benchmark.py --model vit-base-patch16-224 --batch-size 16 --compare-frameworks
"""

import argparse
import time
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
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


class TransformerBenchmark:
    """Generic transformer benchmark class for TVM Relax auto-tuning evaluation."""
    
    def __init__(self, model_name: str, model_type: str, batch_size: int,
                 seq_length: int = None, image_size: int = None,
                 target: str = "cuda", enable_tuning: bool = True):
        self.model_name = model_name
        self.model_type = model_type.lower()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.image_size = image_size
        self.target = tvm.target.Target(target)
        self.enable_tuning = enable_tuning
        
        # Model setup
        self.torch_model, self.tokenizer_or_processor = self._load_model()
        self.torch_model.eval()
        
        # Input data
        self.inputs = self._prepare_inputs()
        
        # TVM setup
        self.relax_mod = None
        self.relax_exec = None
        self.tuning_database = None
        
        logger.info(f"Initialized {model_name} ({model_type}) benchmark: batch_size={batch_size}")
    
    def _load_model(self):
        """Load transformer model based on type."""
        try:
            from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
            import transformers
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        logger.info(f"Loading {self.model_name} ({self.model_type}) from Hugging Face...")
        
        if self.model_type in ["gpt", "gpt2", "gpt-neo", "gpt-j"]:
            from transformers import GPT2Model, GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = GPT2Model.from_pretrained(self.model_name)
            return model, tokenizer
            
        elif self.model_type in ["vit", "vision-transformer"]:
            from transformers import ViTModel, ViTImageProcessor
            processor = ViTImageProcessor.from_pretrained(self.model_name)
            model = ViTModel.from_pretrained(self.model_name)
            return model, processor
            
        elif self.model_type in ["t5", "text-to-text"]:
            from transformers import T5Model, T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            model = T5Model.from_pretrained(self.model_name)
            return model, tokenizer
            
        elif self.model_type in ["roberta", "distilbert"]:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            return model, tokenizer
            
        else:
            # Fallback to auto classes
            tokenizer_or_processor = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            return model, tokenizer_or_processor
    
    def _prepare_inputs(self):
        """Prepare inputs based on model type."""
        if self.model_type in ["gpt", "gpt2", "gpt-neo", "gpt-j", "t5", "roberta", "distilbert"]:
            return self._prepare_text_inputs()
        elif self.model_type in ["vit", "vision-transformer"]:
            return self._prepare_image_inputs()
        else:
            return self._prepare_text_inputs()  # Default to text
    
    def _prepare_text_inputs(self):
        """Prepare text inputs for language models."""
        if self.seq_length is None:
            self.seq_length = 512  # Default sequence length
        
        # Create sample text
        text = "This is a sample input text for transformer model benchmarking and optimization. " * (self.seq_length // 25)
        
        # Tokenize
        encoded = self.tokenizer_or_processor(
            [text] * self.batch_size,
            padding="max_length",
            truncation=True,
            max_length=self.seq_length,
            return_tensors="pt"
        )
        
        return encoded
    
    def _prepare_image_inputs(self):
        """Prepare image inputs for vision models."""
        if self.image_size is None:
            self.image_size = 224  # Default for ViT
        
        # Create random images
        images = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        # Process images
        inputs = self.tokenizer_or_processor(
            images=images,
            return_tensors="pt"
        )
        
        return inputs
    
    def convert_to_relax(self) -> tvm.IRModule:
        """Convert transformer model to TVM Relax IR."""
        logger.info(f"Converting {self.model_type} model to TVM Relax IR...")
        
        with torch.no_grad():
            if self.model_type in ["gpt", "gpt2", "gpt-neo", "gpt-j"]:
                wrapper_class = self._create_gpt_wrapper()
            elif self.model_type in ["vit", "vision-transformer"]:
                wrapper_class = self._create_vit_wrapper()
            elif self.model_type in ["t5"]:
                wrapper_class = self._create_t5_wrapper()
            else:
                wrapper_class = self._create_generic_wrapper()
            
            wrapped_model = wrapper_class(self.torch_model)
            
            # Trace the model
            traced_model = symbolic_trace(wrapped_model)
            
            # Define input info
            input_info = self._get_input_info()
            
            # Convert to Relax
            mod = from_fx(traced_model, input_info)
        
        self.relax_mod = mod
        logger.info(f"{self.model_type} model conversion completed")
        return mod
    
    def _create_gpt_wrapper(self):
        """Create wrapper for GPT-style models."""
        class GPTWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, input_ids, attention_mask):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.last_hidden_state
        
        return GPTWrapper
    
    def _create_vit_wrapper(self):
        """Create wrapper for Vision Transformer models."""
        class ViTWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, pixel_values):
                outputs = self.model(pixel_values=pixel_values)
                return outputs.last_hidden_state
        
        return ViTWrapper
    
    def _create_t5_wrapper(self):
        """Create wrapper for T5 models."""
        class T5Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, input_ids, attention_mask):
                # For T5, we need decoder_input_ids too
                decoder_input_ids = torch.zeros_like(input_ids)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids
                )
                return outputs.last_hidden_state
        
        return T5Wrapper
    
    def _create_generic_wrapper(self):
        """Create generic wrapper for other transformer models."""
        class GenericWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, input_ids, attention_mask):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.last_hidden_state
        
        return GenericWrapper
    
    def _get_input_info(self):
        """Get input info for Relax conversion."""
        if self.model_type in ["vit", "vision-transformer"]:
            return [("pixel_values", (self.batch_size, 3, self.image_size, self.image_size), "float32")]
        elif self.model_type in ["t5"]:
            return [
                ("input_ids", (self.batch_size, self.seq_length), "int64"),
                ("attention_mask", (self.batch_size, self.seq_length), "int64"),
            ]
        else:  # Text models
            return [
                ("input_ids", (self.batch_size, self.seq_length), "int64"),
                ("attention_mask", (self.batch_size, self.seq_length), "int64"),
            ]
    
    def apply_transformer_optimizations(self, mod: tvm.IRModule) -> tvm.IRModule:
        """Apply transformer-specific optimizations."""
        logger.info(f"Applying {self.model_type}-specific optimizations...")
        
        # Base transformer optimizations
        base_passes = [
            relax.transform.DecomposeOpsForInference(),
            relax.transform.CanonicalizeBindings(),
            relax.transform.EliminateCommonSubexpr(),
            relax.transform.CombineParallelMatmul(),  # Critical for multi-head attention
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
        ]
        
        # Model-specific optimizations
        if self.model_type in ["gpt", "gpt2", "gpt-neo", "gpt-j"]:
            # Autoregressive model optimizations
            specific_passes = [
                relax.transform.StaticPlanBlockMemory(),  # Memory optimization for long sequences
                relax.transform.ConvertLayout({
                    "relax.nn.dense": ["NT", "auto"],
                    "relax.nn.batch_matmul": ["NTK", "auto"]
                }),
            ]
        elif self.model_type in ["vit", "vision-transformer"]:
            # Vision model optimizations
            specific_passes = [
                relax.transform.ConvertLayout({
                    "relax.nn.conv2d": ["NHWC", "auto"],
                    "relax.nn.dense": ["NT", "auto"],
                    "relax.nn.batch_matmul": ["NTK", "auto"]
                }),
            ]
        else:
            # Default transformer optimizations
            specific_passes = [
                relax.transform.ConvertLayout({
                    "relax.nn.dense": ["NT", "auto"],
                    "relax.nn.batch_matmul": ["NTK", "auto"]
                }),
            ]
        
        # Combine all passes
        all_passes = base_passes + specific_passes + [
            relax.transform.FoldConstant(),
            relax.transform.LegalizeOps(),
        ]
        
        seq = tvm.transform.Sequential(all_passes)
        optimized_mod = seq(mod)
        
        logger.info(f"{self.model_type} optimizations completed")
        return optimized_mod
    
    def auto_tune_transformer(self, mod: tvm.IRModule, work_dir: str) -> tvm.IRModule:
        """Auto-tune transformer model with architecture-specific strategies."""
        if not self.enable_tuning:
            logger.info("Auto-tuning disabled, skipping...")
            return mod
        
        logger.info(f"Starting {self.model_type} auto-tuning...")
        
        # Create work directory
        os.makedirs(work_dir, exist_ok=True)
        
        # Configure tuning based on model type and size
        num_trials = self._get_tuning_trials()
        
        # Setup meta-schedule database
        database = ms.tir_integration.database.JSONDatabase(
            path_workload=os.path.join(work_dir, "database_workload.json"),
            path_tuning_record=os.path.join(work_dir, "database_tuning_record.json"),
        )
        
        # Architecture-specific tuning configuration
        schedule_rules = self._get_schedule_rules()
        
        # Auto-tuning pipeline
        with ms.Profiler() as profiler:
            tuned_mod = pipeline.static_shape_tuning_pipeline(
                mod=mod,
                target=self.target,
                work_dir=work_dir,
                max_trials_global=num_trials,
                
                # Advanced tuning parameters
                num_trials_per_iter=64,
                max_trials_per_task=256,
                
                # Architecture-optimized schedule rules
                schedule_rules=schedule_rules,
                postprocs=ms.default_config.postproc(target=self.target),
                
                # Tuning strategy
                strategy="evolutionary",
                num_tuning_cores=min(os.cpu_count(), 16),
            )
        
        tuning_time = profiler.get()["total_time"]
        logger.info(f"{self.model_type} auto-tuning completed in {tuning_time:.2f} seconds")
        
        return tuned_mod
    
    def _get_tuning_trials(self) -> int:
        """Get number of tuning trials based on model type and size."""
        base_trials = 1500
        
        if "base" in self.model_name.lower():
            return base_trials
        elif "large" in self.model_name.lower():
            return base_trials * 2
        elif "xl" in self.model_name.lower() or "xxl" in self.model_name.lower():
            return base_trials * 3
        else:
            return base_trials
    
    def _get_schedule_rules(self):
        """Get schedule rules optimized for transformer architectures."""
        if "cuda" not in str(self.target):
            return ms.default_config.schedule_rules(target=self.target)
        
        # CUDA-specific rules optimized for transformers
        return [
            # Multi-level tiling for dense operations
            ms.schedule_rule.MultiLevelTiling(structure="SSSRRSRS"),
            
            # Tensor core optimizations for attention
            ms.schedule_rule.MultiLevelTilingTensorCore(
                intrin_groups=[
                    ms.default_config.tensor_core_intrin_groups()["fp16"]
                ]
            ),
            
            # Auto-inline for better fusion
            ms.schedule_rule.AutoInline(
                into_producer=False,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=True,
            ),
            
            # Cross-thread reduction for attention patterns
            ms.schedule_rule.CrossThreadReduction(
                thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]
            ),
            
            # Vectorization and unrolling
            ms.schedule_rule.ParallelizeVectorizeUnroll(
                max_jobs_per_core=-1,
                max_vectorize_extent=-1,
                unroll_max_steps=[0, 16, 64, 512, 1024],
                unroll_explicit=True,
            ),
        ]
    
    def build_model(self, mod: tvm.IRModule) -> relax.VirtualMachine:
        """Build the optimized transformer model."""
        logger.info(f"Building optimized {self.model_type} model...")
        
        exec = relax.build(mod, target=self.target)
        vm = relax.VirtualMachine(exec, tvm.device(str(self.target).split()[0], 0))
        
        self.relax_exec = exec
        logger.info(f"{self.model_type} model build completed")
        return vm
    
    def benchmark_tvm_relax(self, num_runs: int = 50, warmup_runs: int = 5) -> Dict[str, float]:
        """Benchmark TVM Relax performance."""
        if self.relax_exec is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        logger.info(f"Benchmarking TVM Relax {self.model_type} ({num_runs} runs)...")
        
        # Create VM and prepare inputs
        vm = relax.VirtualMachine(self.relax_exec, tvm.device(str(self.target).split()[0], 0))
        tvm_inputs = self._prepare_tvm_inputs()
        
        # Warmup runs
        for _ in range(warmup_runs):
            if isinstance(tvm_inputs, tuple):
                vm["main"](*tvm_inputs)
            else:
                vm["main"](tvm_inputs)
        
        # Synchronize before timing
        tvm.runtime.Device(str(self.target).split()[0], 0).sync()
        
        # Benchmark runs
        start_time = time.time()
        for _ in range(num_runs):
            if isinstance(tvm_inputs, tuple):
                output = vm["main"](*tvm_inputs)
            else:
                output = vm["main"](tvm_inputs)
        
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
            "model_type": self.model_type
        }
        
        if hasattr(self, 'seq_length') and self.seq_length:
            results["seq_length"] = self.seq_length
        if hasattr(self, 'image_size') and self.image_size:
            results["image_size"] = self.image_size
        
        logger.info(f"TVM Relax {self.model_type} Results: {avg_time:.2f}ms avg latency, {throughput:.1f} samples/sec")
        return results
    
    def _prepare_tvm_inputs(self):
        """Prepare inputs for TVM inference."""
        if self.model_type in ["vit", "vision-transformer"]:
            return tvm.nd.array(self.inputs["pixel_values"].numpy())
        else:  # Text models
            input_ids = tvm.nd.array(self.inputs["input_ids"].numpy())
            attention_mask = tvm.nd.array(self.inputs["attention_mask"].numpy())
            return (input_ids, attention_mask)
    
    def benchmark_huggingface(self, num_runs: int = 50, warmup_runs: int = 5,
                             use_cuda: bool = True) -> Dict[str, float]:
        """Benchmark Hugging Face Transformers for comparison."""
        logger.info(f"Benchmarking Hugging Face {self.model_type} ({num_runs} runs)...")
        
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        model = self.torch_model.to(device)
        
        # Move inputs to device
        inputs_on_device = {}
        for key, value in self.inputs.items():
            if isinstance(value, torch.Tensor):
                inputs_on_device[key] = value.to(device)
            else:
                inputs_on_device[key] = value
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(**inputs_on_device)
        
        if use_cuda:
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(**inputs_on_device)
        
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
            "model_type": self.model_type,
            "device": str(device)
        }
        
        if hasattr(self, 'seq_length') and self.seq_length:
            results["seq_length"] = self.seq_length
        if hasattr(self, 'image_size') and self.image_size:
            results["image_size"] = self.image_size
        
        logger.info(f"Hugging Face {self.model_type} Results: {avg_time:.2f}ms avg latency, {throughput:.1f} samples/sec")
        return results
    
    def run_full_benchmark(self, work_dir: str) -> Dict:
        """Run complete transformer benchmark pipeline."""
        logger.info(f"Starting full {self.model_type} benchmark pipeline...")
        
        results = {
            "model": self.model_name,
            "model_type": self.model_type,
            "batch_size": self.batch_size,
            "target": str(self.target),
            "tuning_enabled": self.enable_tuning,
        }
        
        if hasattr(self, 'seq_length') and self.seq_length:
            results["seq_length"] = self.seq_length
        if hasattr(self, 'image_size') and self.image_size:
            results["image_size"] = self.image_size
        
        try:
            # Convert model
            mod = self.convert_to_relax()
            
            # Apply optimizations
            optimized_mod = self.apply_transformer_optimizations(mod)
            
            # Auto-tune if enabled
            if self.enable_tuning:
                tuned_mod = self.auto_tune_transformer(optimized_mod, work_dir)
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
            
            logger.info(f"{self.model_type} benchmark completed. TVM Relax speedup: {speedup:.2f}x")
            
        except Exception as e:
            logger.error(f"{self.model_type} benchmark failed: {str(e)}")
            results["error"] = str(e)
            raise
        
        return results


def determine_model_type(model_name: str) -> str:
    """Determine model type from model name."""
    model_name_lower = model_name.lower()
    
    if "gpt" in model_name_lower:
        return "gpt"
    elif "vit" in model_name_lower or "vision" in model_name_lower:
        return "vit"
    elif "t5" in model_name_lower:
        return "t5"
    elif "roberta" in model_name_lower:
        return "roberta"
    elif "distilbert" in model_name_lower:
        return "distilbert"
    elif "bert" in model_name_lower:
        return "bert"
    else:
        return "generic"


def main():
    parser = argparse.ArgumentParser(description="Transformer Models TVM Relax Auto-Tuning Benchmark")
    parser.add_argument("--model", default="gpt2", 
                       help="Transformer model name from Hugging Face")
    parser.add_argument("--model-type", default=None,
                       help="Model type (gpt, vit, t5, bert, etc.). Auto-detected if not specified")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=None, help="Sequence length (for text models)")
    parser.add_argument("--image-size", type=int, default=None, help="Image size (for vision models)")
    parser.add_argument("--target", default="cuda", help="TVM target (cuda, cpu, etc.)")
    parser.add_argument("--tune", action="store_true", help="Enable auto-tuning")
    parser.add_argument("--compare-frameworks", action="store_true", 
                       help="Compare with other frameworks")
    parser.add_argument("--work-dir", default="./transformer_tune_logs", 
                       help="Directory for tuning logs")
    parser.add_argument("--output", default="transformer_benchmark_results.json", 
                       help="Output file for results")
    parser.add_argument("--num-runs", type=int, default=50, 
                       help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Auto-detect model type if not specified
    model_type = args.model_type or determine_model_type(args.model)
    
    # Set default parameters based on model type
    if model_type in ["vit", "vision-transformer"] and args.image_size is None:
        args.image_size = 224
    elif model_type in ["gpt", "bert", "t5"] and args.seq_length is None:
        args.seq_length = 512
    
    # Create benchmark
    benchmark = TransformerBenchmark(
        model_name=args.model,
        model_type=model_type,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        image_size=args.image_size,
        target=args.target,
        enable_tuning=args.tune
    )
    
    # Set work directory with model-specific name
    work_dir = args.work_dir.replace("transformer", model_type)
    
    # Run benchmark
    results = benchmark.run_full_benchmark(work_dir)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print(f"{model_type.upper()} TRANSFORMER BENCHMARK SUMMARY")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Model Type: {model_type}")
    print(f"Batch Size: {args.batch_size}")
    if args.seq_length:
        print(f"Sequence Length: {args.seq_length}")
    if args.image_size:
        print(f"Image Size: {args.image_size}")
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
    
    if "speedup" in results:
        print(f"\nTVM Relax Speedup: {results['speedup']:.2f}x vs Hugging Face")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()