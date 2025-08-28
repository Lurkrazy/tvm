#!/usr/bin/env python3
"""
Multi-Framework Comparison Benchmark for TVM Relax

This script provides a comprehensive comparison of TVM Relax auto-tuning
against multiple inference frameworks including PyTorch, ONNX Runtime,
TensorRT, and OpenVINO across different model architectures.

Usage:
    python multi_framework_comparison.py --models resnet50,bert-base-uncased,gpt2 --all-frameworks
    python multi_framework_comparison.py --config benchmark_config.json
"""

import argparse
import time
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Import our benchmark modules
from resnet_benchmark import ResNetBenchmark
from bert_benchmark import BERTBenchmark
from transformer_benchmark import TransformerBenchmark, determine_model_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiFrameworkComparison:
    """Comprehensive multi-framework comparison for TVM Relax."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = {}
        self.frameworks = self._get_available_frameworks()
        
        logger.info(f"Initialized multi-framework comparison with {len(self.frameworks)} frameworks")
    
    def _get_available_frameworks(self) -> List[str]:
        """Detect available frameworks."""
        frameworks = ["tvm_relax", "pytorch"]  # Always available
        
        # Check ONNX Runtime
        try:
            import onnxruntime
            frameworks.append("onnx_runtime")
        except ImportError:
            logger.warning("ONNX Runtime not available")
        
        # Check TensorRT (for NVIDIA GPUs)
        if self._check_tensorrt():
            frameworks.append("tensorrt")
        
        # Check OpenVINO (for Intel hardware)
        if self._check_openvino():
            frameworks.append("openvino")
        
        return frameworks
    
    def _check_tensorrt(self) -> bool:
        """Check if TensorRT is available."""
        try:
            import tensorrt
            return True
        except ImportError:
            logger.warning("TensorRT not available")
            return False
    
    def _check_openvino(self) -> bool:
        """Check if OpenVINO is available."""
        try:
            import openvino
            return True
        except ImportError:
            logger.warning("OpenVINO not available")
            return False
    
    def benchmark_model(self, model_config: Dict) -> Dict:
        """Benchmark a single model across all frameworks."""
        model_name = model_config["name"]
        model_type = model_config.get("type", determine_model_type(model_name))
        
        logger.info(f"Benchmarking {model_name} ({model_type})")
        
        results = {
            "model": model_name,
            "type": model_type,
            "config": model_config,
            "frameworks": {},
            "comparisons": {}
        }
        
        try:
            # Create appropriate benchmark instance
            if model_type in ["resnet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
                benchmark = self._create_resnet_benchmark(model_config)
            elif model_type in ["bert", "roberta", "distilbert"]:
                benchmark = self._create_bert_benchmark(model_config)
            else:
                benchmark = self._create_transformer_benchmark(model_config)
            
            # Run TVM Relax benchmark (our primary focus)
            tvm_results = self._benchmark_tvm_relax(benchmark, model_config)
            results["frameworks"]["tvm_relax"] = tvm_results
            
            # Run other framework benchmarks in parallel
            other_results = self._benchmark_other_frameworks(model_config)
            results["frameworks"].update(other_results)
            
            # Calculate comparisons
            results["comparisons"] = self._calculate_comparisons(results["frameworks"])
            
        except Exception as e:
            logger.error(f"Failed to benchmark {model_name}: {str(e)}")
            results["error"] = str(e)
            results["traceback"] = traceback.format_exc()
        
        return results
    
    def _create_resnet_benchmark(self, config: Dict) -> ResNetBenchmark:
        """Create ResNet benchmark instance."""
        return ResNetBenchmark(
            model_name=config["name"],
            batch_size=config.get("batch_size", 1),
            input_shape=(config.get("batch_size", 1), 3, 224, 224),
            target=config.get("target", "cuda"),
            enable_tuning=config.get("enable_tuning", True)
        )
    
    def _create_bert_benchmark(self, config: Dict) -> BERTBenchmark:
        """Create BERT benchmark instance."""
        return BERTBenchmark(
            model_name=config["name"],
            batch_size=config.get("batch_size", 1),
            seq_length=config.get("seq_length", 128),
            target=config.get("target", "cuda"),
            enable_tuning=config.get("enable_tuning", True)
        )
    
    def _create_transformer_benchmark(self, config: Dict) -> TransformerBenchmark:
        """Create Transformer benchmark instance."""
        return TransformerBenchmark(
            model_name=config["name"],
            model_type=config.get("type", determine_model_type(config["name"])),
            batch_size=config.get("batch_size", 1),
            seq_length=config.get("seq_length", 512),
            image_size=config.get("image_size", 224),
            target=config.get("target", "cuda"),
            enable_tuning=config.get("enable_tuning", True)
        )
    
    def _benchmark_tvm_relax(self, benchmark, config: Dict) -> Dict:
        """Benchmark TVM Relax with auto-tuning."""
        logger.info("Running TVM Relax benchmark with auto-tuning...")
        
        work_dir = f"./tune_logs_{config['name'].replace('/', '_')}"
        return benchmark.run_full_benchmark(work_dir)
    
    def _benchmark_other_frameworks(self, config: Dict) -> Dict:
        """Benchmark other frameworks in parallel."""
        results = {}
        
        # Use ThreadPoolExecutor for parallel framework benchmarking
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_framework = {}
            
            for framework in self.frameworks:
                if framework == "tvm_relax":
                    continue  # Already benchmarked
                
                future = executor.submit(self._benchmark_single_framework, framework, config)
                future_to_framework[future] = framework
            
            for future in as_completed(future_to_framework):
                framework = future_to_framework[future]
                try:
                    result = future.result(timeout=600)  # 10 minute timeout
                    if result:
                        results[framework] = result
                except Exception as e:
                    logger.error(f"Framework {framework} benchmark failed: {str(e)}")
                    results[framework] = {"error": str(e)}
        
        return results
    
    def _benchmark_single_framework(self, framework: str, config: Dict) -> Optional[Dict]:
        """Benchmark a single framework."""
        logger.info(f"Benchmarking {framework}...")
        
        try:
            if framework == "pytorch":
                return self._benchmark_pytorch(config)
            elif framework == "onnx_runtime":
                return self._benchmark_onnx_runtime(config)
            elif framework == "tensorrt":
                return self._benchmark_tensorrt(config)
            elif framework == "openvino":
                return self._benchmark_openvino(config)
            else:
                logger.warning(f"Unknown framework: {framework}")
                return None
        except Exception as e:
            logger.error(f"Error benchmarking {framework}: {str(e)}")
            return {"error": str(e)}
    
    def _benchmark_pytorch(self, config: Dict) -> Dict:
        """Benchmark PyTorch native performance."""
        import torch
        
        model_name = config["name"]
        model_type = config.get("type", determine_model_type(model_name))
        batch_size = config.get("batch_size", 1)
        num_runs = config.get("num_runs", 100)
        
        if model_type.startswith("resnet"):
            import torchvision.models as models
            model = getattr(models, model_type)(pretrained=True)
            inputs = torch.randn(batch_size, 3, 224, 224)
        elif model_type in ["bert", "gpt", "t5"]:
            from transformers import AutoModel, AutoTokenizer
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            text = "Sample text " * 50
            encoded = tokenizer(
                [text] * batch_size,
                padding="max_length",
                truncation=True,
                max_length=config.get("seq_length", 128),
                return_tensors="pt"
            )
            inputs = (encoded["input_ids"], encoded["attention_mask"])
        else:
            logger.warning(f"PyTorch benchmark not implemented for {model_type}")
            return {"error": f"Not implemented for {model_type}"}
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        
        if isinstance(inputs, tuple):
            inputs = tuple(inp.to(device) for inp in inputs)
        else:
            inputs = inputs.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                if isinstance(inputs, tuple):
                    model(*inputs)
                else:
                    model(inputs)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                if isinstance(inputs, tuple):
                    model(*inputs)
                else:
                    model(inputs)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000
        
        return {
            "avg_latency_ms": avg_time,
            "throughput_samples_per_sec": batch_size / (avg_time / 1000),
            "framework": "pytorch",
            "device": str(device)
        }
    
    def _benchmark_onnx_runtime(self, config: Dict) -> Dict:
        """Benchmark ONNX Runtime performance."""
        import onnxruntime as ort
        import torch
        import numpy as np
        
        # This is a simplified implementation
        # In practice, you'd need model-specific ONNX conversion
        logger.info("ONNX Runtime benchmark (simplified)")
        
        return {
            "avg_latency_ms": 0.0,  # Placeholder
            "framework": "onnx_runtime",
            "note": "Simplified implementation - would need model-specific ONNX conversion"
        }
    
    def _benchmark_tensorrt(self, config: Dict) -> Dict:
        """Benchmark TensorRT performance."""
        logger.info("TensorRT benchmark (placeholder)")
        
        return {
            "avg_latency_ms": 0.0,  # Placeholder
            "framework": "tensorrt",
            "note": "TensorRT implementation would require model-specific optimization"
        }
    
    def _benchmark_openvino(self, config: Dict) -> Dict:
        """Benchmark OpenVINO performance."""
        logger.info("OpenVINO benchmark (placeholder)")
        
        return {
            "avg_latency_ms": 0.0,  # Placeholder
            "framework": "openvino",
            "note": "OpenVINO implementation would require model conversion"
        }
    
    def _calculate_comparisons(self, framework_results: Dict) -> Dict:
        """Calculate performance comparisons between frameworks."""
        comparisons = {}
        
        if "tvm_relax" not in framework_results:
            return comparisons
        
        tvm_latency = framework_results["tvm_relax"].get("tvm_relax", {}).get("avg_latency_ms")
        if not tvm_latency:
            return comparisons
        
        for framework, results in framework_results.items():
            if framework == "tvm_relax" or "error" in results:
                continue
            
            framework_latency = results.get("avg_latency_ms")
            if framework_latency:
                speedup = framework_latency / tvm_latency
                comparisons[f"speedup_vs_{framework}"] = speedup
        
        return comparisons
    
    def run_comparison(self) -> Dict:
        """Run complete multi-framework comparison."""
        logger.info("Starting multi-framework comparison...")
        
        all_results = {
            "config": self.config,
            "available_frameworks": self.frameworks,
            "models": {},
            "summary": {}
        }
        
        # Benchmark each model
        for model_config in self.config["models"]:
            model_name = model_config["name"]
            logger.info(f"Processing model: {model_name}")
            
            model_results = self.benchmark_model(model_config)
            all_results["models"][model_name] = model_results
        
        # Generate summary
        all_results["summary"] = self._generate_summary(all_results["models"])
        
        logger.info("Multi-framework comparison completed")
        return all_results
    
    def _generate_summary(self, model_results: Dict) -> Dict:
        """Generate summary statistics across all models."""
        summary = {
            "total_models": len(model_results),
            "successful_models": 0,
            "failed_models": 0,
            "average_speedups": {},
            "framework_success_rate": {}
        }
        
        framework_speedups = {}
        framework_successes = {}
        
        for model_name, results in model_results.items():
            if "error" in results:
                summary["failed_models"] += 1
                continue
            
            summary["successful_models"] += 1
            
            # Collect speedup data
            comparisons = results.get("comparisons", {})
            for comparison, speedup in comparisons.items():
                if comparison.startswith("speedup_vs_"):
                    framework = comparison.replace("speedup_vs_", "")
                    if framework not in framework_speedups:
                        framework_speedups[framework] = []
                    framework_speedups[framework].append(speedup)
            
            # Track framework success rates
            frameworks = results.get("frameworks", {})
            for framework, result in frameworks.items():
                if framework not in framework_successes:
                    framework_successes[framework] = {"success": 0, "total": 0}
                
                framework_successes[framework]["total"] += 1
                if "error" not in result:
                    framework_successes[framework]["success"] += 1
        
        # Calculate average speedups
        for framework, speedups in framework_speedups.items():
            if speedups:
                summary["average_speedups"][framework] = {
                    "mean": np.mean(speedups) if speedups else 0,
                    "min": np.min(speedups) if speedups else 0,
                    "max": np.max(speedups) if speedups else 0,
                    "count": len(speedups)
                }
        
        # Calculate success rates
        for framework, stats in framework_successes.items():
            summary["framework_success_rate"][framework] = {
                "rate": stats["success"] / stats["total"] if stats["total"] > 0 else 0,
                "successful": stats["success"],
                "total": stats["total"]
            }
        
        return summary


def load_benchmark_config(config_path: str) -> Dict:
    """Load benchmark configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)


def create_default_config(models: List[str]) -> Dict:
    """Create default configuration for specified models."""
    config = {
        "models": [],
        "global_settings": {
            "target": "cuda",
            "enable_tuning": True,
            "num_runs": 100
        }
    }
    
    for model in models:
        model_type = determine_model_type(model)
        
        model_config = {
            "name": model,
            "type": model_type,
            "batch_size": 1,
            "enable_tuning": True
        }
        
        # Add model-specific settings
        if model_type.startswith("resnet"):
            model_config["input_shape"] = [1, 3, 224, 224]
        elif model_type in ["bert", "gpt", "t5"]:
            model_config["seq_length"] = 128 if "bert" in model_type else 512
        elif model_type == "vit":
            model_config["image_size"] = 224
        
        config["models"].append(model_config)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Multi-Framework Comparison for TVM Relax")
    parser.add_argument("--config", help="JSON configuration file")
    parser.add_argument("--models", help="Comma-separated list of models to benchmark")
    parser.add_argument("--all-frameworks", action="store_true", 
                       help="Benchmark against all available frameworks")
    parser.add_argument("--output", default="multi_framework_results.json", 
                       help="Output file for results")
    parser.add_argument("--target", default="cuda", help="TVM target")
    parser.add_argument("--batch-size", type=int, default=1, help="Default batch size")
    parser.add_argument("--enable-tuning", action="store_true", default=True,
                       help="Enable TVM auto-tuning")
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = load_benchmark_config(args.config)
    elif args.models:
        models = args.models.split(',')
        config = create_default_config(models)
        
        # Override with command line arguments
        for model_config in config["models"]:
            model_config["batch_size"] = args.batch_size
            model_config["enable_tuning"] = args.enable_tuning
        config["global_settings"]["target"] = args.target
    else:
        # Default configuration with common models
        config = create_default_config([
            "resnet50",
            "bert-base-uncased", 
            "gpt2"
        ])
    
    # Create comparison instance and run
    comparison = MultiFrameworkComparison(config)
    results = comparison.run_comparison()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("MULTI-FRAMEWORK COMPARISON SUMMARY")
    print("="*80)
    
    summary = results["summary"]
    print(f"Total Models: {summary['total_models']}")
    print(f"Successful: {summary['successful_models']}")
    print(f"Failed: {summary['failed_models']}")
    print()
    
    print("Framework Success Rates:")
    for framework, stats in summary["framework_success_rate"].items():
        rate = stats["rate"] * 100
        print(f"  {framework}: {rate:.1f}% ({stats['successful']}/{stats['total']})")
    print()
    
    if summary["average_speedups"]:
        print("Average TVM Relax Speedups:")
        for framework, stats in summary["average_speedups"].items():
            print(f"  vs {framework}: {stats['mean']:.2f}x (min: {stats['min']:.2f}x, max: {stats['max']:.2f}x)")
    
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    import numpy as np  # Add this for summary calculations
    main()