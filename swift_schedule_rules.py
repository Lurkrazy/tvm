#!/usr/bin/env python3
"""
Swift-style Schedule Rules for TVM MetaSchedule

Custom schedule rules implementing Swift-style heuristics for high parallelism
and optimal CUDA SM occupancy.
"""

import logging
from typing import List, Optional, Dict, Any
import math

try:
    import tvm
    from tvm import meta_schedule as ms
    from tvm.meta_schedule.schedule_rule import ScheduleRule, ReuseType
    from tvm.tir.schedule import Schedule, BlockRV
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
    Schedule = object
    BlockRV = object

logger = logging.getLogger(__name__)


class SwiftHighParallelismRule:
    """
    Swift-style rule factory that creates rules optimized for high parallelism.
    
    Key characteristics:
    - Prioritizes many small thread blocks (2-4 per SM)
    - Uses SSSRRSRS tiling pattern
    - Targets 64-256 threads per block
    - Enables cooperative shared memory loads
    """
    
    @staticmethod
    def create_tensorcore_rule(
        target: str = "cuda",
        max_threads_per_block: int = 256,
        target_blocks_per_sm: int = 4
    ) -> Optional[ScheduleRule]:
        """Create TensorCore-optimized rule with Swift characteristics."""
        if not TVM_AVAILABLE:
            return None
            
        try:
            # Common TensorCore intrinsic groups for different precisions
            intrin_groups = [
                # FP16 tensor cores (most common)
                {
                    "compute": "wmma.m16n16k16.f16.f16.f32.compute",
                    "load_a": "wmma.m16n16k16.f16.f16.f32.load_a",
                    "load_b": "wmma.m16n16k16.f16.f16.f32.load_b", 
                    "store": "wmma.m16n16k16.f16.f16.f32.store"
                },
                # Mixed precision alternatives
                {
                    "compute": "mma.m16n8k16.f16.f16.f32.compute",
                    "load_a": "mma.m16n8k16.f16.f16.f32.load_a",
                    "load_b": "mma.m16n8k16.f16.f16.f32.load_b",
                    "store": "mma.m16n8k16.f16.f16.f32.store"
                },
                # INT8 tensor cores
                {
                    "compute": "dp4a.int8.int8.int32.compute",
                }
            ]
            
            rule = ms.schedule_rule.MultiLevelTilingTensorCore(
                intrin_groups=intrin_groups,
                structure="SSSRRSRS",  # Swift: prioritize spatial over reduction
                tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
                max_innermost_factor=64,  # Bounded for register pressure
                vector_load_lens=[1, 2, 4, 8, 16],  # Progressive vectorization
                reuse_read=ReuseType(req="may", levels=[1, 2], scope="shared"),
                reuse_write=ReuseType(req="may", levels=[2], scope="shared"),
                use_software_pipeline=True,  # Enable ping-pong buffering
            )
            
            logger.info("Created Swift TensorCore rule")
            return rule
            
        except Exception as e:
            logger.warning(f"Failed to create TensorCore rule: {e}")
            return None
    
    @staticmethod
    def create_simt_rule(
        target: str = "cuda",
        max_threads_per_block: int = 256
    ) -> Optional[ScheduleRule]:
        """Create SIMT-optimized rule with Swift characteristics."""
        if not TVM_AVAILABLE:
            return None
            
        try:
            def swift_filter(sch: Schedule, block_rv: BlockRV) -> bool:
                """Filter for compute-intensive blocks suitable for Swift tiling."""
                try:
                    block = sch.get(block_rv)
                    name = block.name_hint.lower()
                    
                    # Target matrix operations and convolutions
                    compute_ops = ['matmul', 'dense', 'conv2d', 'batch_matmul', 'gemm']
                    is_compute = any(op in name for op in compute_ops)
                    
                    # Avoid cache/staging blocks
                    avoid_ops = ['cache', 'stage', 'temp', 'pad']
                    is_avoided = any(op in name for op in avoid_ops)
                    
                    return is_compute and not is_avoided
                    
                except Exception:
                    return False
            
            rule = ms.schedule_rule.MultiLevelTiling(
                structure="SSSRRSRS",  # Swift pattern: spatial-heavy
                tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
                max_innermost_factor=64,
                vector_load_lens=[1, 2, 4, 8, 16],
                reuse_read=ReuseType(req="may", levels=[1, 2], scope="shared"),
                reuse_write=ReuseType(req="may", levels=[2], scope="shared"),
                filter_fn=swift_filter,
            )
            
            logger.info("Created Swift SIMT rule")
            return rule
            
        except Exception as e:
            logger.warning(f"Failed to create SIMT rule: {e}")
            return None


class SwiftOccupancyRule:
    """
    Rule factory focused on maximizing SM occupancy through resource management.
    """
    
    @staticmethod
    def estimate_occupancy(
        threads_per_block: int,
        shared_mem_per_block: int,
        registers_per_thread: int,
        target: str = "cuda"
    ) -> float:
        """Estimate theoretical occupancy based on resource usage."""
        
        # Default SM limits (can be refined based on target architecture)
        sm_limits = {
            "max_threads_per_sm": 2048,  # Common for modern GPUs
            "max_blocks_per_sm": 32,
            "max_shared_mem_per_sm": 49152,  # 48KB
            "max_registers_per_sm": 65536,
        }
        
        try:
            # Extract architecture from target if possible
            if "sm_" in target:
                arch = target.split("sm_")[1].split()[0]
                arch_num = int(arch)
                
                # Update limits based on architecture
                if arch_num >= 80:  # Ampere+
                    sm_limits["max_shared_mem_per_sm"] = 102400  # 100KB
                elif arch_num >= 70:  # Volta+
                    sm_limits["max_shared_mem_per_sm"] = 98304   # 96KB
                    
        except Exception:
            pass  # Use defaults
        
        # Calculate limits (handle zero values gracefully)
        blocks_by_threads = sm_limits["max_threads_per_sm"] // max(threads_per_block, 1)
        blocks_by_shared_mem = sm_limits["max_shared_mem_per_sm"] // max(shared_mem_per_block, 1)
        blocks_by_registers = sm_limits["max_registers_per_sm"] // max(registers_per_thread * threads_per_block, 1)
        
        # Actual blocks per SM is minimum of all constraints
        blocks_per_sm = min(
            blocks_by_threads,
            blocks_by_shared_mem, 
            blocks_by_registers,
            sm_limits["max_blocks_per_sm"]
        )
        
        # Occupancy = (actual threads) / (max possible threads)
        actual_threads = blocks_per_sm * threads_per_block
        occupancy = actual_threads / sm_limits["max_threads_per_sm"]
        
        return min(occupancy, 1.0)
    
    @staticmethod
    def create_occupancy_aware_rule() -> Optional[ScheduleRule]:
        """Create a rule that considers occupancy in tiling decisions."""
        if not TVM_AVAILABLE:
            return None
            
        # This would be a custom rule that analyzes block sizes and resource usage
        # For now, return a conservative multi-level tiling rule
        try:
            rule = ms.schedule_rule.MultiLevelTiling(
                structure="SSSRRSRS",
                tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
                max_innermost_factor=32,  # Conservative for occupancy
                vector_load_lens=[1, 2, 4, 8],  # Moderate vectorization
                reuse_read=ReuseType(req="may", levels=[1], scope="shared"),  # Light shared mem use
            )
            
            logger.info("Created occupancy-aware rule")
            return rule
            
        except Exception as e:
            logger.warning(f"Failed to create occupancy-aware rule: {e}")
            return None


class SwiftVectorizationRule:
    """
    Rule factory for aggressive but controlled vectorization following Swift principles.
    """
    
    @staticmethod
    def create_vectorization_rule(
        vector_lengths: List[int] = [1, 2, 4, 8, 16],
        max_unroll_steps: List[int] = [16, 64, 512]
    ) -> Optional[ScheduleRule]:
        """Create vectorization-focused rule."""
        if not TVM_AVAILABLE:
            return None
            
        try:
            rule = ms.schedule_rule.ParallelizeVectorizeUnroll(
                max_jobs_per_core=16,  # High parallelism
                max_vectorize_extent=max(vector_lengths),
                unroll_max_steps=max_unroll_steps,
                unroll_explicit=True,  # Explicit unrolling for better control
            )
            
            logger.info(f"Created vectorization rule with lengths {vector_lengths}")
            return rule
            
        except Exception as e:
            logger.warning(f"Failed to create vectorization rule: {e}")
            return None


class SwiftReductionRule:
    """
    Rule factory for efficient cross-thread reductions.
    """
    
    @staticmethod
    def create_reduction_rule(
        thread_extents: List[int] = [4, 8, 16, 32, 64, 128, 256]
    ) -> Optional[ScheduleRule]:
        """Create cross-thread reduction rule."""
        if not TVM_AVAILABLE:
            return None
            
        try:
            # Cross-thread reduction for warp and block level reductions
            reduction_rule = ms.schedule_rule.CrossThreadReduction(
                thread_extents=thread_extents
            )
            
            logger.info(f"Created reduction rule with thread extents {thread_extents}")
            return reduction_rule
            
        except Exception as e:
            logger.warning(f"Failed to create reduction rule: {e}")
            return None
    
    @staticmethod 
    def create_rfactor_rule() -> Optional[ScheduleRule]:
        """Create RFactor rule for reduction optimization."""
        if not TVM_AVAILABLE:
            return None
            
        try:
            rfactor_rule = ms.schedule_rule.AddRFactor(
                max_jobs_per_core=16,  # High parallelism
                max_innermost_factor=64,  # Bounded inner factors
            )
            
            logger.info("Created RFactor rule")
            return rfactor_rule
            
        except Exception as e:
            logger.warning(f"Failed to create RFactor rule: {e}")
            return None


def create_swift_schedule_rules(
    target: str = "cuda",
    enable_tensorcore: bool = True,
    max_threads_per_block: int = 256
) -> List[ScheduleRule]:
    """
    Create a complete set of Swift-style schedule rules.
    
    Parameters
    ----------
    target : str
        Target architecture string
    enable_tensorcore : bool
        Whether to include TensorCore rules
    max_threads_per_block : int
        Maximum threads per thread block
        
    Returns
    -------
    List[ScheduleRule]
        List of schedule rules for Swift space generation
    """
    if not TVM_AVAILABLE:
        return []
        
    rules = []
    
    # High-priority tensor core rule
    if enable_tensorcore:
        tc_rule = SwiftHighParallelismRule.create_tensorcore_rule(
            target=target,
            max_threads_per_block=max_threads_per_block
        )
        if tc_rule:
            rules.append(tc_rule)
    
    # SIMT fallback rule
    simt_rule = SwiftHighParallelismRule.create_simt_rule(
        target=target,
        max_threads_per_block=max_threads_per_block
    )
    if simt_rule:
        rules.append(simt_rule)
    
    # Occupancy-aware rule
    occ_rule = SwiftOccupancyRule.create_occupancy_aware_rule()
    if occ_rule:
        rules.append(occ_rule)
    
    # Vectorization rule
    vec_rule = SwiftVectorizationRule.create_vectorization_rule()
    if vec_rule:
        rules.append(vec_rule)
    
    # Reduction rules
    reduction_rule = SwiftReductionRule.create_reduction_rule()
    if reduction_rule:
        rules.append(reduction_rule)
        
    rfactor_rule = SwiftReductionRule.create_rfactor_rule()
    if rfactor_rule:
        rules.append(rfactor_rule)
    
    # Auto-inline for small operations
    try:
        inline_rule = ms.schedule_rule.AutoInline(
            into_producer=False,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=False,
            require_injective=True,
            require_ordered=True,
            disallow_op=["tir.exp"]  # Avoid expensive ops
        )
        rules.append(inline_rule)
    except Exception as e:
        logger.warning(f"Failed to create AutoInline rule: {e}")
    
    # Auto-bind for thread assignment
    try:
        bind_rule = ms.schedule_rule.AutoBind(
            max_threadblock=max_threads_per_block,
            thread_extents=[32, 64, 128, 256, 512]
        )
        rules.append(bind_rule)
    except Exception as e:
        logger.warning(f"Failed to create AutoBind rule: {e}")
    
    logger.info(f"Created {len(rules)} Swift schedule rules")
    return rules


def main():
    """Test the Swift schedule rules."""
    logging.basicConfig(level=logging.INFO)
    
    if not TVM_AVAILABLE:
        print("TVM not available. Cannot test schedule rules.")
        return
    
    rules = create_swift_schedule_rules(target="cuda -arch=sm_80")
    
    print(f"Created {len(rules)} Swift schedule rules:")
    for i, rule in enumerate(rules):
        print(f"  {i+1}. {type(rule).__name__}")
    
    # Test occupancy estimation
    occ = SwiftOccupancyRule.estimate_occupancy(
        threads_per_block=256,
        shared_mem_per_block=8192,  # 8KB
        registers_per_thread=32,
        target="cuda -arch=sm_80"
    )
    print(f"Estimated occupancy: {occ:.2%}")


if __name__ == "__main__":
    main()