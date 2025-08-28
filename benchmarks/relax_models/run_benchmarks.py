#!/usr/bin/env python3
"""
Quick Start Script for TVM Relax Model Benchmarks

This script provides easy access to run various benchmark scenarios
for evaluating TVM Relax auto-tuning performance.

Usage:
    python run_benchmarks.py --quick
    python run_benchmarks.py --comprehensive
    python run_benchmarks.py --custom
"""

import argparse
import subprocess
import sys
import os
import json
import time
from typing import List, Dict


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ {description} failed with error: {str(e)}")
        return False


def quick_benchmark():
    """Run quick benchmarks for demonstration."""
    print("🚀 Running Quick TVM Relax Benchmarks")
    print("This will run lightweight benchmarks for demonstration...")
    
    benchmarks = [
        {
            "cmd": ["python", "resnet_benchmark.py", "--model", "resnet18", 
                   "--batch-size", "1", "--tune", "--num-runs", "20"],
            "desc": "ResNet18 Auto-Tuning Demo"
        },
        {
            "cmd": ["python", "bert_benchmark.py", "--model", "distilbert-base-uncased", 
                   "--seq-length", "64", "--tune", "--num-runs", "20"],
            "desc": "DistilBERT Auto-Tuning Demo"
        },
        {
            "cmd": ["python", "transformer_benchmark.py", "--model", "gpt2", 
                   "--seq-length", "256", "--tune", "--num-runs", "20"],
            "desc": "GPT-2 Auto-Tuning Demo"
        }
    ]
    
    results = []
    for benchmark in benchmarks:
        success = run_command(benchmark["cmd"], benchmark["desc"])
        results.append({"name": benchmark["desc"], "success": success})
    
    # Summary
    print(f"\n{'='*60}")
    print("QUICK BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for result in results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"{result['name']}: {status}")


def comprehensive_benchmark():
    """Run comprehensive benchmarks for full evaluation."""
    print("🔬 Running Comprehensive TVM Relax Benchmarks")
    print("This will run full benchmarks with comparison to other frameworks...")
    
    benchmarks = [
        {
            "cmd": ["python", "resnet_benchmark.py", "--model", "resnet50", 
                   "--batch-size", "8", "--tune", "--compare-frameworks"],
            "desc": "ResNet50 Comprehensive Benchmark"
        },
        {
            "cmd": ["python", "bert_benchmark.py", "--model", "bert-base-uncased", 
                   "--seq-length", "128", "--tune", "--compare-frameworks"],
            "desc": "BERT-Base Comprehensive Benchmark"
        },
        {
            "cmd": ["python", "transformer_benchmark.py", "--model", "gpt2", 
                   "--seq-length", "512", "--tune", "--compare-frameworks"],
            "desc": "GPT-2 Comprehensive Benchmark"
        },
        {
            "cmd": ["python", "multi_framework_comparison.py", 
                   "--models", "resnet50,bert-base-uncased,gpt2", 
                   "--all-frameworks"],
            "desc": "Multi-Framework Comparison"
        }
    ]
    
    results = []
    start_time = time.time()
    
    for benchmark in benchmarks:
        success = run_command(benchmark["cmd"], benchmark["desc"])
        results.append({"name": benchmark["desc"], "success": success})
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print()
    for result in results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"{result['name']}: {status}")


def custom_benchmark():
    """Run custom benchmark based on user configuration."""
    print("⚙️  Running Custom TVM Relax Benchmarks")
    
    config_file = "benchmark_config.json"
    if not os.path.exists(config_file):
        print(f"❌ Configuration file {config_file} not found!")
        print("Creating a sample configuration file...")
        
        # Create sample config
        sample_config = {
            "models": [
                {
                    "name": "resnet50",
                    "type": "resnet",
                    "batch_size": 4,
                    "enable_tuning": True
                },
                {
                    "name": "bert-base-uncased",
                    "type": "bert",
                    "batch_size": 2,
                    "seq_length": 256,
                    "enable_tuning": True
                }
            ],
            "global_settings": {
                "target": "cuda",
                "num_runs": 50,
                "compare_frameworks": True
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        print(f"✅ Created sample configuration: {config_file}")
        print("Please edit the configuration file and run again.")
        return
    
    # Run custom benchmark
    cmd = ["python", "multi_framework_comparison.py", "--config", config_file]
    success = run_command(cmd, "Custom Benchmark from Configuration")
    
    if success:
        print("✅ Custom benchmark completed successfully!")
    else:
        print("❌ Custom benchmark failed!")


def batch_size_analysis():
    """Run batch size analysis for different models."""
    print("📊 Running Batch Size Analysis")
    
    models = [
        {"name": "resnet50", "script": "resnet_benchmark.py"},
        {"name": "bert-base-uncased", "script": "bert_benchmark.py", "extra": ["--seq-length", "128"]},
    ]
    
    batch_sizes = [1, 2, 4, 8, 16]
    
    for model in models:
        print(f"\n🔍 Analyzing {model['name']} across batch sizes...")
        
        for bs in batch_sizes:
            cmd = ["python", model["script"], "--model", model["name"], 
                   "--batch-size", str(bs), "--tune", "--num-runs", "30",
                   "--output", f"{model['name']}_bs{bs}_results.json"]
            
            if "extra" in model:
                cmd.extend(model["extra"])
            
            description = f"{model['name']} Batch Size {bs}"
            run_command(cmd, description)
    
    print("\n📈 Batch size analysis completed!")
    print("Check individual result files for detailed analysis.")


def target_analysis():
    """Run analysis across different targets."""
    print("🎯 Running Target Analysis")
    
    targets = ["cuda", "llvm", "llvm -mcpu=skylake-avx512"]
    model = "resnet18"  # Use smaller model for multi-target testing
    
    for target in targets:
        target_name = target.replace(" ", "_").replace("-", "_")
        cmd = ["python", "resnet_benchmark.py", "--model", model,
               "--target", target, "--tune", "--num-runs", "30",
               "--output", f"{model}_{target_name}_results.json"]
        
        description = f"{model} on {target}"
        run_command(cmd, description)
    
    print("\n🎯 Target analysis completed!")


def main():
    parser = argparse.ArgumentParser(description="TVM Relax Benchmark Runner")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick demo benchmarks")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive benchmarks with framework comparison")
    parser.add_argument("--custom", action="store_true",
                       help="Run custom benchmarks from configuration file")
    parser.add_argument("--batch-analysis", action="store_true",
                       help="Run batch size analysis")
    parser.add_argument("--target-analysis", action="store_true",
                       help="Run target comparison analysis")
    parser.add_argument("--all", action="store_true",
                       help="Run all benchmark types")
    
    args = parser.parse_args()
    
    # Change to benchmark directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("🔥 TVM Relax Auto-Tuning Benchmark Suite")
    print(f"📁 Working directory: {os.getcwd()}")
    
    if args.all:
        print("🚀 Running ALL benchmark types...")
        quick_benchmark()
        comprehensive_benchmark()
        custom_benchmark()
        batch_size_analysis()
        target_analysis()
    elif args.quick:
        quick_benchmark()
    elif args.comprehensive:
        comprehensive_benchmark()
    elif args.custom:
        custom_benchmark()
    elif args.batch_analysis:
        batch_size_analysis()
    elif args.target_analysis:
        target_analysis()
    else:
        print("Please specify a benchmark type:")
        print("  --quick: Quick demo benchmarks")
        print("  --comprehensive: Full benchmarks with framework comparison")
        print("  --custom: Custom benchmarks from configuration")
        print("  --batch-analysis: Batch size optimization analysis")
        print("  --target-analysis: Target comparison analysis")
        print("  --all: Run all benchmark types")
        print()
        print("Example: python run_benchmarks.py --quick")


if __name__ == "__main__":
    main()