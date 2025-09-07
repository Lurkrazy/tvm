#!/usr/bin/env python3
"""
Swift-style Space Generator for TVM MetaSchedule

This module implements a Swift-style space generator that prioritizes high parallelism
and SM occupancy on CUDA GPUs, with support for Tensor Core acceleration when available.
"""

import logging
import sys
import os
from typing import List, Optional, Dict, Any, Callable

# Add TVM to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

try:
    import tvm
    from tvm import meta_schedule as ms
    from tvm.meta_schedule.schedule_rule import ScheduleRule, ReuseType
    from tvm.meta_schedule.postproc import Postproc
    from tvm.meta_schedule.mutator import Mutator
    from tvm.meta_schedule.space_generator import SpaceGenerator
    from tvm.contrib.nvcc import have_tensorcore
    from tvm.target import Target
    TVM_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    TVM_AVAILABLE = False
    print(f"TVM not available: {e}")
    
    # Create placeholder classes for graceful fallback
    class Target:
        def __init__(self, target_str):
            self.attrs = {}
    
    class ReuseType:
        def __init__(self, req, levels, scope):
            self.req = req
            self.levels = levels
            self.scope = scope
            
        def as_dict(self):
            return {"req": self.req, "levels": self.levels, "scope": self.scope}
    
    # Placeholder types for annotations
    ScheduleRule = object
    Postproc = object
    Mutator = object
    SpaceGenerator = object

logger = logging.getLogger(__name__)


def discover_available_apis():
    """Discover and log available TVM MetaSchedule APIs."""
    if not TVM_AVAILABLE:
        return {}
    
    capability_table = {
        "schedule_rules": [],
        "postprocs": [],
        "mutators": [],
        "tensor_core_support": False,
        "space_generator_methods": []
    }
    
    print("=== TVM MetaSchedule API Discovery ===")
    
    # Discover ScheduleRule classes
    print("\n--- Schedule Rules ---")
    for attr_name in dir(ms.schedule_rule):
        if not attr_name.startswith('_'):
            attr = getattr(ms.schedule_rule, attr_name, None)
            if hasattr(attr, '__name__') and 'Rule' in attr.__name__:
                capability_table["schedule_rules"].append(attr_name)
                print(f"✓ {attr_name}")
    
    # Discover Postproc classes
    print("\n--- Postprocessors ---")
    for attr_name in dir(ms.postproc):
        if not attr_name.startswith('_'):
            attr = getattr(ms.postproc, attr_name, None)
            if hasattr(attr, '__name__') and 'Postproc' in attr.__name__:
                capability_table["postprocs"].append(attr_name)
                print(f"✓ {attr_name}")
    
    # Discover Mutator classes  
    print("\n--- Mutators ---")
    for attr_name in dir(ms.mutator):
        if not attr_name.startswith('_'):
            attr = getattr(ms.mutator, attr_name, None)
            if hasattr(attr, '__name__') and 'Mutator' in attr.__name__:
                capability_table["mutators"].append(attr_name)
                print(f"✓ {attr_name}")
    
    # Check SpaceGenerator methods
    print("\n--- SpaceGenerator Methods ---")
    for attr_name in dir(ms.SpaceGenerator):
        if not attr_name.startswith('_'):
            capability_table["space_generator_methods"].append(attr_name)
            print(f"✓ {attr_name}")
    
    # Check TensorCore support functions
    print("\n--- TensorCore Support ---")
    try:
        # Try to check if we have TensorCore detection
        capability_table["tensor_core_support"] = hasattr(tvm.contrib.nvcc, 'have_tensorcore')
        print(f"✓ TensorCore detection: {capability_table['tensor_core_support']}")
    except Exception as e:
        print(f"✗ TensorCore detection failed: {e}")
    
    return capability_table


def swift_block_filter(sch, block_rv):
    """
    Filter function for blocks that should use Swift-style tiling.
    Targets compute-intensive blocks: matmul, dense, batch_matmul, conv2d.
    """
    try:
        block = sch.get(block_rv)
        block_name = block.name_hint.lower()
        
        # Target high-intensity compute kernels
        compute_patterns = [
            'matmul', 'dense', 'batch_matmul', 'bmm',
            'conv2d', 'conv1d', 'conv3d', 
            'gemm', 'dot', 'compute'
        ]
        
        # Exclude cache/temporary blocks
        exclude_patterns = ['cache', 'temp', 'pad', 'reshape', 'transpose']
        
        # Check if block matches compute patterns and doesn't match exclude patterns
        is_compute_block = any(pattern in block_name for pattern in compute_patterns)
        is_excluded = any(pattern in block_name for pattern in exclude_patterns)
        
        result = is_compute_block and not is_excluded
        
        if result:
            logger.info(f"Swift filter: Accepting block '{block_name}' for tiling")
        else:
            logger.debug(f"Swift filter: Skipping block '{block_name}'")
            
        return result
        
    except Exception as e:
        logger.warning(f"Swift filter error on block: {e}")
        return False


class SwiftSpaceGenerator:
    """
    Factory for creating Swift-style space generators optimized for CUDA.
    
    The Swift approach prioritizes:
    1. High parallelism with 2-4 thread blocks per SM
    2. Tensor Core acceleration when available
    3. Cooperative fetch for shared memory staging
    4. Vectorization and bounded unroll
    5. Resource-aware tile sizing
    """
    
    def __init__(self, target_str: str = "cuda"):
        self.target_str = target_str
        self.capabilities = discover_available_apis() if TVM_AVAILABLE else {}
        self.tensor_core_available = self._check_tensor_core_support()
        
    def _check_tensor_core_support(self) -> bool:
        """Check if Tensor Core intrinsics are available."""
        if not TVM_AVAILABLE:
            return False
            
        try:
            target = Target(self.target_str)
            if 'cuda' not in self.target_str:
                return False
                
            # Check if we have TensorCore detection capability
            if self.capabilities.get("tensor_core_support", False):
                return have_tensorcore(target=target)
            else:
                # Fallback: assume modern GPUs have TensorCore if arch >= sm_70
                if "arch" in target.attrs:
                    arch = target.attrs["arch"]
                    if arch.startswith("sm_"):
                        version = int(arch[3:])
                        return version >= 70
                        
        except Exception as e:
            logger.warning(f"TensorCore check failed: {e}")
            
        return False
    
    def _get_swift_schedule_rules(self) -> List[ScheduleRule]:
        """Build Swift-style schedule rules prioritizing high parallelism."""
        if not TVM_AVAILABLE:
            return []
            
        rules = []
        
        # High-priority: TensorCore path if available
        if self.tensor_core_available and "MultiLevelTilingTensorCore" in self.capabilities["schedule_rules"]:
            logger.info("Adding TensorCore-optimized tiling rule")
            
            # Swift-style structure: SSSRRSRS (CTA -> warp -> thread, with reduction)
            # Use software pipeline for high arithmetic intensity
            try:
                tc_rule = ms.schedule_rule.MultiLevelTilingTensorCore(
                    intrin_groups=[
                        # Common tensor core intrinsics for different dtypes
                        {"dp4a": "dp4a.int8.int8.int32"},
                        {"wmma": "wmma.m16n16k16.f16.f16.f32"},
                        {"mma": "mma.m16n8k16.f16.f16.f32"},
                    ],
                    structure="SSSRRSRS",  # Swift-style: prioritize many thread blocks
                    tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
                    max_innermost_factor=64,  # Bounded for register pressure
                    vector_load_lens=[1, 2, 4, 8, 16],  # Progressive vectorization
                    reuse_read=ReuseType(req="may", levels=[1, 2], scope="shared"),
                    reuse_write=ReuseType(req="may", levels=[2], scope="shared"),
                    use_software_pipeline=True,  # Enable double buffering
                )
                rules.append(tc_rule)
            except Exception as e:
                logger.warning(f"Failed to create TensorCore rule: {e}")
        
        # Fallback: Generic multi-level tiling with Swift characteristics
        if "MultiLevelTiling" in self.capabilities["schedule_rules"]:
            logger.info("Adding generic multi-level tiling rule")
            try:
                generic_rule = ms.schedule_rule.MultiLevelTiling(
                    structure="SSSRRSRS",  # Swift pattern
                    tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
                    max_innermost_factor=64,
                    vector_load_lens=[1, 2, 4, 8, 16],
                    reuse_read=ReuseType(req="may", levels=[1, 2], scope="shared"),
                    reuse_write=ReuseType(req="may", levels=[2], scope="shared"),
                    filter_fn=swift_block_filter,
                )
                rules.append(generic_rule)
            except Exception as e:
                logger.warning(f"Failed to create generic tiling rule: {e}")
        
        # Auto-inline for small operations
        if "AutoInline" in self.capabilities["schedule_rules"]:
            try:
                rules.append(ms.schedule_rule.AutoInline(
                    into_producer=False,
                    into_consumer=True,
                    inline_const_tensor=True,
                    disallow_if_then_else=False,
                    require_injective=True,
                    require_ordered=True,
                    disallow_op=["tir.exp"]
                ))
            except Exception as e:
                logger.warning(f"Failed to create AutoInline rule: {e}")
        
        # Cross-thread reduction for reductions
        if "CrossThreadReduction" in self.capabilities["schedule_rules"]:
            try:
                rules.append(ms.schedule_rule.CrossThreadReduction(
                    thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]
                ))
            except Exception as e:
                logger.warning(f"Failed to create CrossThreadReduction rule: {e}")
        
        # Add RFactor for reduction optimization
        if "AddRFactor" in self.capabilities["schedule_rules"]:
            try:
                rules.append(ms.schedule_rule.AddRFactor(
                    max_jobs_per_core=16,
                    max_innermost_factor=64
                ))
            except Exception as e:
                logger.warning(f"Failed to create AddRFactor rule: {e}")
                
        # Parallelize, vectorize, unroll
        if "ParallelizeVectorizeUnroll" in self.capabilities["schedule_rules"]:
            try:
                rules.append(ms.schedule_rule.ParallelizeVectorizeUnroll(
                    max_jobs_per_core=16,
                    max_vectorize_extent=64,
                    unroll_max_steps=[0, 16, 64, 512],  # Bounded unroll
                    unroll_explicit=True
                ))
            except Exception as e:
                logger.warning(f"Failed to create ParallelizeVectorizeUnroll rule: {e}")
        
        # Auto-bind thread axes
        if "AutoBind" in self.capabilities["schedule_rules"]:
            try:
                rules.append(ms.schedule_rule.AutoBind(
                    max_threadblock=1024,
                    thread_extents=[32, 64, 128, 256, 512, 1024]
                ))
            except Exception as e:
                logger.warning(f"Failed to create AutoBind rule: {e}")
        
        logger.info(f"Created {len(rules)} schedule rules for Swift space")
        return rules
    
    def _get_swift_postprocessors(self) -> List[Postproc]:
        """Build Swift-style postprocessors for GPU optimization."""
        if not TVM_AVAILABLE:
            return []
            
        postprocs = []
        
        # Cooperative fetch rewrite for shared memory staging
        if "RewriteCooperativeFetch" in self.capabilities["postprocs"]:
            try:
                postprocs.append(ms.postproc.RewriteCooperativeFetch())
                logger.info("Added cooperative fetch rewriter")
            except Exception as e:
                logger.warning(f"Failed to create RewriteCooperativeFetch: {e}")
        
        # Parallelize-vectorize-unroll rewrite
        if "RewriteParallelVectorizeUnroll" in self.capabilities["postprocs"]:
            try:
                postprocs.append(ms.postproc.RewriteParallelVectorizeUnroll())
                logger.info("Added parallel-vectorize-unroll rewriter")
            except Exception as e:
                logger.warning(f"Failed to create RewriteParallelVectorizeUnroll: {e}")
        
        # Tensorization rewrite (if TensorCore available)
        if self.tensor_core_available and "RewriteTensorize" in self.capabilities["postprocs"]:
            try:
                postprocs.append(ms.postproc.RewriteTensorize())
                logger.info("Added tensorization rewriter")
            except Exception as e:
                logger.warning(f"Failed to create RewriteTensorize: {e}")
        
        # Unbound block rewrite
        if "RewriteUnboundBlock" in self.capabilities["postprocs"]:
            try:
                postprocs.append(ms.postproc.RewriteUnboundBlock())
                logger.info("Added unbound block rewriter")
            except Exception as e:
                logger.warning(f"Failed to create RewriteUnboundBlock: {e}")
        
        # GPU code verification (critical for CUDA)
        if "VerifyGPUCode" in self.capabilities["postprocs"]:
            try:
                postprocs.append(ms.postproc.VerifyGPUCode())
                logger.info("Added GPU code verifier")
            except Exception as e:
                logger.warning(f"Failed to create VerifyGPUCode: {e}")
        
        # Disallow dynamic loops (CUDA requirement)
        if "DisallowDynamicLoop" in self.capabilities["postprocs"]:
            try:
                postprocs.append(ms.postproc.DisallowDynamicLoop())
                logger.info("Added dynamic loop disabler")
            except Exception as e:
                logger.warning(f"Failed to create DisallowDynamicLoop: {e}")
        
        logger.info(f"Created {len(postprocs)} postprocessors for Swift space")
        return postprocs
    
    def _get_swift_mutator_probs(self) -> Dict[Mutator, float]:
        """Build Swift-style mutator probabilities for CUDA optimization."""
        if not TVM_AVAILABLE:
            return {}
            
        mutator_probs = {}
        
        try:
            if self.tensor_core_available:
                # Use CUDA TensorCore mutators if available
                mutator_probs = Mutator.create("cuda-tensorcore")
                logger.info("Using CUDA TensorCore mutator configuration")
            else:
                # Fallback to regular CUDA mutators
                mutator_probs = Mutator.create("cuda")
                logger.info("Using CUDA mutator configuration")
                
        except Exception as e:
            logger.warning(f"Failed to create mutator configuration: {e}")
            # Manual fallback
            try:
                mutator_probs = {
                    ms.mutator.MutateTileSize(): 0.9,  # Heavy tile size mutations
                    ms.mutator.MutateUnroll(): 0.08,   # Light unroll mutations
                    ms.mutator.MutateThreadBinding(): 0.02,  # Minimal thread binding changes
                }
                logger.info("Using manual mutator configuration")
            except Exception as e2:
                logger.warning(f"Manual mutator creation also failed: {e2}")
        
        return mutator_probs
    
    def create_space_generator(self) -> Optional[SpaceGenerator]:
        """Create a Swift-style space generator."""
        if not TVM_AVAILABLE:
            logger.error("TVM not available, cannot create space generator")
            return None
        
        logger.info("Creating Swift-style space generator...")
        
        # Get components
        sch_rules = self._get_swift_schedule_rules()
        postprocs = self._get_swift_postprocessors()
        mutator_probs = self._get_swift_mutator_probs()
        
        # Print capability summary
        print(f"\n=== Swift Space Generator Summary ===")
        print(f"Target: {self.target_str}")
        print(f"TensorCore support: {self.tensor_core_available}")
        print(f"Schedule rules: {len(sch_rules)}")
        print(f"Postprocessors: {len(postprocs)}")
        print(f"Mutator types: {len(mutator_probs)}")
        
        # Create the space generator
        try:
            # Try PostOrderApply constructor
            space_gen = ms.space_generator.PostOrderApply(
                f_block_filter=swift_block_filter,
                sch_rules=sch_rules,
                postprocs=postprocs,
                mutator_probs=mutator_probs
            )
            logger.info("Successfully created Swift space generator")
            return space_gen
            
        except Exception as e:
            logger.error(f"Failed to create space generator: {e}")
            
            # Try fallback with SpaceGenerator.create
            try:
                space_gen = ms.SpaceGenerator.create(
                    kind="post-order-apply",
                    f_block_filter=swift_block_filter,
                    sch_rules=sch_rules,
                    postprocs=postprocs,
                    mutator_probs=mutator_probs
                )
                logger.info("Successfully created Swift space generator (fallback)")
                return space_gen
            except Exception as e2:
                logger.error(f"Fallback space generator creation failed: {e2}")
                return None


def main():
    """Main function for standalone testing."""
    logging.basicConfig(level=logging.INFO)
    
    if not TVM_AVAILABLE:
        print("TVM not available. Cannot proceed with space generator creation.")
        return
    
    # Test the Swift space generator
    factory = SwiftSpaceGenerator(target_str="cuda -arch=sm_80")
    space_gen = factory.create_space_generator()
    
    if space_gen:
        print("✓ Swift space generator created successfully!")
    else:
        print("✗ Failed to create Swift space generator")


if __name__ == "__main__":
    main()