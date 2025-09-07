#!/usr/bin/env python3
"""
Swift-style Postprocessors for TVM MetaSchedule

Custom postprocessors implementing Swift-style optimizations for CUDA kernels,
focusing on cooperative memory access patterns and GPU verification.
"""

import logging
from typing import List, Optional, Dict, Any

try:
    import tvm
    from tvm import meta_schedule as ms
    from tvm.meta_schedule.postproc import Postproc
    from tvm.target import Target
    TVM_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    TVM_AVAILABLE = False
    print(f"TVM not available: {e}")
    
    # Create placeholder classes for graceful fallback
    class Target:
        def __init__(self, target_str):
            self.attrs = {}
    
    # Placeholder types for annotations  
    Postproc = object

logger = logging.getLogger(__name__)


class SwiftCooperativeMemoryOpts:
    """
    Factory for cooperative memory access optimizations.
    """
    
    @staticmethod
    def create_cooperative_fetch() -> Optional[Postproc]:
        """Create cooperative fetch rewriter for shared memory staging."""
        if not TVM_AVAILABLE:
            return None
            
        try:
            postproc = ms.postproc.RewriteCooperativeFetch()
            logger.info("Created cooperative fetch postprocessor")
            return postproc
        except Exception as e:
            logger.warning(f"Failed to create cooperative fetch postprocessor: {e}")
            return None
    
    @staticmethod
    def create_vectorization_rewriter() -> Optional[Postproc]:
        """Create parallel-vectorize-unroll rewriter."""
        if not TVM_AVAILABLE:
            return None
            
        try:
            postproc = ms.postproc.RewriteParallelVectorizeUnroll()
            logger.info("Created parallel-vectorize-unroll postprocessor")
            return postproc
        except Exception as e:
            logger.warning(f"Failed to create vectorization rewriter: {e}")
            return None


class SwiftTensorCoreOpts:
    """
    Factory for Tensor Core specific optimizations.
    """
    
    @staticmethod
    def create_tensorize_rewriter() -> Optional[Postproc]:
        """Create tensorization rewriter for tensor core intrinsics."""
        if not TVM_AVAILABLE:
            return None
            
        try:
            postproc = ms.postproc.RewriteTensorize()
            logger.info("Created tensorization postprocessor")
            return postproc
        except Exception as e:
            logger.warning(f"Failed to create tensorization rewriter: {e}")
            return None


class SwiftGPUVerification:
    """
    Factory for GPU-specific verification and safety checks.
    """
    
    @staticmethod
    def create_gpu_verifier() -> Optional[Postproc]:
        """Create GPU code verifier."""
        if not TVM_AVAILABLE:
            return None
            
        try:
            postproc = ms.postproc.VerifyGPUCode()
            logger.info("Created GPU code verifier")
            return postproc
        except Exception as e:
            logger.warning(f"Failed to create GPU verifier: {e}")
            return None
    
    @staticmethod
    def create_dynamic_loop_disallower() -> Optional[Postproc]:
        """Create dynamic loop disallower (required for CUDA)."""
        if not TVM_AVAILABLE:
            return None
            
        try:
            postproc = ms.postproc.DisallowDynamicLoop()
            logger.info("Created dynamic loop disallower")
            return postproc
        except Exception as e:
            logger.warning(f"Failed to create dynamic loop disallower: {e}")
            return None


class SwiftBlockOptimizations:
    """
    Factory for block-level optimizations.
    """
    
    @staticmethod
    def create_unbound_block_rewriter() -> Optional[Postproc]:
        """Create unbound block rewriter."""
        if not TVM_AVAILABLE:
            return None
            
        try:
            postproc = ms.postproc.RewriteUnboundBlock()
            logger.info("Created unbound block rewriter")
            return postproc
        except Exception as e:
            logger.warning(f"Failed to create unbound block rewriter: {e}")
            return None
    
    @staticmethod
    def create_reduction_block_rewriter() -> Optional[Postproc]:
        """Create reduction block rewriter."""
        if not TVM_AVAILABLE:
            return None
            
        try:
            postproc = ms.postproc.RewriteReductionBlock()
            logger.info("Created reduction block rewriter")
            return postproc
        except Exception as e:
            logger.warning(f"Failed to create reduction block rewriter: {e}")
            return None


class SwiftLayoutOptimizations:
    """
    Factory for layout and memory pattern optimizations.
    """
    
    @staticmethod
    def create_layout_rewriter() -> Optional[Postproc]:
        """Create layout rewriter for optimal memory access patterns."""
        if not TVM_AVAILABLE:
            return None
            
        try:
            postproc = ms.postproc.RewriteLayout()
            logger.info("Created layout rewriter")
            return postproc
        except Exception as e:
            logger.warning(f"Failed to create layout rewriter: {e}")
            return None


def create_swift_postprocessors(
    target: str = "cuda",
    enable_tensorcore: bool = True,
    strict_verification: bool = True
) -> List[Postproc]:
    """
    Create a complete set of Swift-style postprocessors.
    
    Parameters
    ----------
    target : str
        Target architecture string
    enable_tensorcore : bool
        Whether to include TensorCore-specific postprocessors
    strict_verification : bool
        Whether to enable strict GPU verification
        
    Returns
    -------
    List[Postproc]
        List of postprocessors for Swift space generation
    """
    if not TVM_AVAILABLE:
        return []
        
    postprocs = []
    
    # Phase 1: Memory access optimizations (high priority)
    coop_fetch = SwiftCooperativeMemoryOpts.create_cooperative_fetch()
    if coop_fetch:
        postprocs.append(coop_fetch)
    
    vec_rewriter = SwiftCooperativeMemoryOpts.create_vectorization_rewriter()
    if vec_rewriter:
        postprocs.append(vec_rewriter)
    
    # Phase 2: Tensor Core optimizations (if enabled)
    if enable_tensorcore:
        tensorize = SwiftTensorCoreOpts.create_tensorize_rewriter()
        if tensorize:
            postprocs.append(tensorize)
    
    # Phase 3: Block-level optimizations
    unbound_rewriter = SwiftBlockOptimizations.create_unbound_block_rewriter()
    if unbound_rewriter:
        postprocs.append(unbound_rewriter)
    
    reduction_rewriter = SwiftBlockOptimizations.create_reduction_block_rewriter()
    if reduction_rewriter:
        postprocs.append(reduction_rewriter)
    
    # Phase 4: Layout optimizations
    layout_rewriter = SwiftLayoutOptimizations.create_layout_rewriter()
    if layout_rewriter:
        postprocs.append(layout_rewriter)
    
    # Phase 5: GPU verification and safety (critical for CUDA)
    if "cuda" in target.lower():
        # Dynamic loop disallowance (required for CUDA)
        dynamic_loop_disallower = SwiftGPUVerification.create_dynamic_loop_disallower()
        if dynamic_loop_disallower:
            postprocs.append(dynamic_loop_disallower)
        
        # GPU code verification
        if strict_verification:
            gpu_verifier = SwiftGPUVerification.create_gpu_verifier()
            if gpu_verifier:
                postprocs.append(gpu_verifier)
    
    logger.info(f"Created {len(postprocs)} Swift postprocessors")
    return postprocs


class SwiftPostprocessorSequence:
    """
    Manager for the Swift postprocessor sequence with phase-aware ordering.
    """
    
    def __init__(self, target: str = "cuda", enable_tensorcore: bool = True):
        self.target = target
        self.enable_tensorcore = enable_tensorcore
        self.phases = {
            "memory_opt": [],
            "compute_opt": [],
            "block_opt": [],
            "layout_opt": [],
            "verification": []
        }
        self._build_phases()
    
    def _build_phases(self):
        """Build postprocessor phases in order."""
        if not TVM_AVAILABLE:
            return
        
        # Phase 1: Memory access patterns
        self.phases["memory_opt"] = [
            SwiftCooperativeMemoryOpts.create_cooperative_fetch(),
            SwiftCooperativeMemoryOpts.create_vectorization_rewriter(),
        ]
        
        # Phase 2: Compute optimizations
        if self.enable_tensorcore:
            self.phases["compute_opt"] = [
                SwiftTensorCoreOpts.create_tensorize_rewriter(),
            ]
        
        # Phase 3: Block optimizations
        self.phases["block_opt"] = [
            SwiftBlockOptimizations.create_unbound_block_rewriter(),
            SwiftBlockOptimizations.create_reduction_block_rewriter(),
        ]
        
        # Phase 4: Layout optimizations
        self.phases["layout_opt"] = [
            SwiftLayoutOptimizations.create_layout_rewriter(),
        ]
        
        # Phase 5: Verification (CUDA-specific)
        if "cuda" in self.target.lower():
            self.phases["verification"] = [
                SwiftGPUVerification.create_dynamic_loop_disallower(),
                SwiftGPUVerification.create_gpu_verifier(),
            ]
        
        # Filter out None values
        for phase_name, postprocs in self.phases.items():
            self.phases[phase_name] = [p for p in postprocs if p is not None]
    
    def get_ordered_postprocessors(self) -> List[Postproc]:
        """Get postprocessors in optimal phase order."""
        ordered = []
        
        # Add phases in order
        phase_order = ["memory_opt", "compute_opt", "block_opt", "layout_opt", "verification"]
        
        for phase_name in phase_order:
            ordered.extend(self.phases[phase_name])
        
        return ordered
    
    def get_phase_summary(self) -> Dict[str, int]:
        """Get summary of postprocessors per phase."""
        return {phase: len(postprocs) for phase, postprocs in self.phases.items()}


def create_ordered_swift_postprocessors(
    target: str = "cuda",
    enable_tensorcore: bool = True
) -> List[Postproc]:
    """
    Create Swift postprocessors in optimal phase order.
    
    This function ensures postprocessors are applied in the correct sequence
    for maximum effectiveness.
    """
    if not TVM_AVAILABLE:
        return []
    
    sequence = SwiftPostprocessorSequence(target=target, enable_tensorcore=enable_tensorcore)
    postprocs = sequence.get_ordered_postprocessors()
    
    logger.info(f"Created {len(postprocs)} ordered Swift postprocessors")
    logger.info(f"Phase breakdown: {sequence.get_phase_summary()}")
    
    return postprocs


def main():
    """Test the Swift postprocessors."""
    logging.basicConfig(level=logging.INFO)
    
    if not TVM_AVAILABLE:
        print("TVM not available. Cannot test postprocessors.")
        return
    
    # Test basic postprocessor creation
    postprocs = create_swift_postprocessors(target="cuda -arch=sm_80")
    
    print(f"Created {len(postprocs)} Swift postprocessors:")
    for i, postproc in enumerate(postprocs):
        print(f"  {i+1}. {type(postproc).__name__}")
    
    # Test ordered postprocessor sequence
    print("\n--- Ordered Postprocessor Sequence ---")
    ordered_postprocs = create_ordered_swift_postprocessors(target="cuda -arch=sm_80")
    
    print(f"Created {len(ordered_postprocs)} ordered postprocessors:")
    for i, postproc in enumerate(ordered_postprocs):
        print(f"  {i+1}. {type(postproc).__name__}")
    
    # Test phase summary
    sequence = SwiftPostprocessorSequence(target="cuda -arch=sm_80")
    summary = sequence.get_phase_summary()
    
    print(f"\nPhase summary: {summary}")


if __name__ == "__main__":
    main()