#!/usr/bin/env python3
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

"""
Helper functions to dump CUDA PTX and verify tensor core intrinsics.
"""

import logging
import re
from typing import Optional, List, Dict, Any
from pathlib import Path

import tvm


def extract_ptx_from_module(module: tvm.runtime.Module) -> Optional[str]:
    """Extract PTX source code from a TVM runtime module.
    
    Args:
        module: TVM runtime module
        
    Returns:
        PTX source code as string, or None if not found
    """
    # Try direct PTX extraction
    try:
        ptx_source = module.get_source("ptx")
        if ptx_source:
            return ptx_source
    except Exception:
        pass
    
    # Try to get source without format specification
    try:
        source = module.get_source()
        if source and ".version" in source:  # PTX signature
            return source
    except Exception:
        pass
    
    # Check imported modules
    if hasattr(module, 'imported_modules'):
        for imported_mod in module.imported_modules:
            try:
                ptx_source = imported_mod.get_source("ptx")
                if ptx_source:
                    return ptx_source
            except Exception:
                pass
            
            try:
                source = imported_mod.get_source()
                if source and ".version" in source:
                    return source
            except Exception:
                pass
    
    return None


def extract_cuda_source_from_module(module: tvm.runtime.Module) -> Optional[str]:
    """Extract CUDA C source code from a TVM runtime module.
    
    Args:
        module: TVM runtime module
        
    Returns:
        CUDA source code as string, or None if not found
    """
    try:
        cuda_source = module.get_source("cuda")
        if cuda_source:
            return cuda_source
    except Exception:
        pass
    
    # Check imported modules
    if hasattr(module, 'imported_modules'):
        for imported_mod in module.imported_modules:
            try:
                cuda_source = imported_mod.get_source("cuda")
                if cuda_source:
                    return cuda_source
            except Exception:
                pass
    
    return None


def analyze_ptx_for_tensor_cores(ptx_source: str) -> Dict[str, Any]:
    """Analyze PTX source for tensor core usage and intrinsics.
    
    Args:
        ptx_source: PTX source code
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "has_tensor_cores": False,
        "wmma_instructions": [],
        "mma_instructions": [],
        "tensor_core_types": set(),
        "compute_capability": None,
        "shared_memory_usage": [],
        "register_usage": None
    }
    
    lines = ptx_source.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Check for compute capability
        if ".target" in line and "sm_" in line:
            match = re.search(r'sm_(\d+)', line)
            if match:
                analysis["compute_capability"] = int(match.group(1))
        
        # Check for WMMA instructions (Volta/Turing tensor cores)
        if "wmma." in line:
            analysis["has_tensor_cores"] = True
            analysis["wmma_instructions"].append(line)
            
            # Extract data types
            if "f16" in line:
                analysis["tensor_core_types"].add("fp16")
            if "f32" in line:
                analysis["tensor_core_types"].add("fp32")
            if "s8" in line or "u8" in line:
                analysis["tensor_core_types"].add("int8")
        
        # Check for MMA instructions (Ampere+ tensor cores)
        if "mma." in line and ("sync" in line or "async" in line):
            analysis["has_tensor_cores"] = True
            analysis["mma_instructions"].append(line)
            
            # Extract data types
            if "f16" in line:
                analysis["tensor_core_types"].add("fp16")
            if "bf16" in line:
                analysis["tensor_core_types"].add("bf16")
            if "f32" in line:
                analysis["tensor_core_types"].add("fp32")
            if "s8" in line or "u8" in line:
                analysis["tensor_core_types"].add("int8")
        
        # Check for shared memory usage
        if ".shared" in line and ".align" in line:
            analysis["shared_memory_usage"].append(line)
    
    # Convert set to list for JSON serialization
    analysis["tensor_core_types"] = list(analysis["tensor_core_types"])
    
    return analysis


def dump_cuda_artifacts(
    executable: tvm.runtime.vm.Executable,
    output_dir: str,
    analyze_tensor_cores: bool = True
) -> Dict[str, Any]:
    """Dump CUDA artifacts (PTX, CUDA source) and analyze them.
    
    Args:
        executable: VM executable
        output_dir: Output directory
        analyze_tensor_cores: Whether to analyze for tensor core usage
        
    Returns:
        Dictionary with dump results and analysis
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "ptx_dumped": False,
        "cuda_dumped": False,
        "analysis": None
    }
    
    module = executable.mod
    
    # Extract and save PTX
    ptx_source = extract_ptx_from_module(module)
    if ptx_source:
        ptx_file = output_path / "cuda_kernel.ptx"
        with open(ptx_file, 'w') as f:
            f.write(ptx_source)
        logging.info(f"Saved PTX source to {ptx_file}")
        results["ptx_dumped"] = True
        
        # Analyze PTX for tensor cores
        if analyze_tensor_cores:
            analysis = analyze_ptx_for_tensor_cores(ptx_source)
            results["analysis"] = analysis
            
            # Save analysis
            import json
            analysis_file = output_path / "ptx_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            logging.info(f"Saved PTX analysis to {analysis_file}")
            
            # Log tensor core findings
            if analysis["has_tensor_cores"]:
                logging.info("✓ Tensor core instructions found in PTX!")
                logging.info(f"  - Tensor core types: {analysis['tensor_core_types']}")
                logging.info(f"  - WMMA instructions: {len(analysis['wmma_instructions'])}")
                logging.info(f"  - MMA instructions: {len(analysis['mma_instructions'])}")
            else:
                logging.info("⚠ No tensor core instructions found in PTX")
    else:
        logging.warning("Could not extract PTX source")
    
    # Extract and save CUDA source
    cuda_source = extract_cuda_source_from_module(module)
    if cuda_source:
        cuda_file = output_path / "cuda_kernel.cu"
        with open(cuda_file, 'w') as f:
            f.write(cuda_source)
        logging.info(f"Saved CUDA source to {cuda_file}")
        results["cuda_dumped"] = True
    else:
        logging.warning("Could not extract CUDA source")
    
    return results


def verify_tensor_core_usage(analysis: Dict[str, Any]) -> None:
    """Print a verification report for tensor core usage.
    
    Args:
        analysis: Analysis results from analyze_ptx_for_tensor_cores
    """
    print("\n" + "="*60)
    print("TENSOR CORE VERIFICATION REPORT")
    print("="*60)
    
    if analysis["compute_capability"]:
        sm = analysis["compute_capability"]
        print(f"Compute Capability: SM_{sm}")
        
        # Check if tensor cores are supported
        if sm >= 70:  # Volta and later
            print("✓ Hardware supports tensor cores")
        else:
            print("✗ Hardware does not support tensor cores")
    
    if analysis["has_tensor_cores"]:
        print("✓ TENSOR CORES ARE BEING USED")
        print(f"  Data types: {', '.join(analysis['tensor_core_types'])}")
        
        if analysis["wmma_instructions"]:
            print(f"  WMMA instructions: {len(analysis['wmma_instructions'])}")
        
        if analysis["mma_instructions"]:
            print(f"  MMA instructions: {len(analysis['mma_instructions'])}")
            
        print("\nExample instructions:")
        all_instructions = analysis["wmma_instructions"] + analysis["mma_instructions"]
        for i, instr in enumerate(all_instructions[:3]):  # Show first 3
            print(f"  {i+1}. {instr}")
        
        if len(all_instructions) > 3:
            print(f"  ... and {len(all_instructions) - 3} more")
    else:
        print("✗ NO TENSOR CORE INSTRUCTIONS FOUND")
        print("  This may indicate:")
        print("    - The workload doesn't benefit from tensor cores")
        print("    - MetaSchedule didn't select tensor core schedules")
        print("    - The model shapes are not suitable for tensor cores")
    
    if analysis["shared_memory_usage"]:
        print(f"\nShared memory allocations: {len(analysis['shared_memory_usage'])}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze PTX file for tensor core usage")
    parser.add_argument("ptx_file", help="Path to PTX file")
    
    args = parser.parse_args()
    
    with open(args.ptx_file, 'r') as f:
        ptx_source = f.read()
    
    analysis = analyze_ptx_for_tensor_cores(ptx_source)
    verify_tensor_core_usage(analysis)