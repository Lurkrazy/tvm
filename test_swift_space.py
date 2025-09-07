#!/usr/bin/env python3
"""
Unit tests for Swift-style Space Generator

Tests for API discovery, component creation, and basic functionality
without requiring a full TVM build.
"""

import unittest
import logging
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import Swift components
from swift_space_generator import SwiftSpaceGenerator, discover_available_apis, swift_block_filter
from swift_schedule_rules import (
    SwiftHighParallelismRule, SwiftOccupancyRule, 
    SwiftVectorizationRule, SwiftReductionRule,
    create_swift_schedule_rules
)
from swift_postprocs import (
    SwiftCooperativeMemoryOpts, SwiftTensorCoreOpts,
    SwiftGPUVerification, SwiftBlockOptimizations,
    SwiftPostprocessorSequence, create_swift_postprocessors
)

# Mock TVM components for testing without TVM installation
class MockTVM:
    """Mock TVM module for testing."""
    
    class meta_schedule:
        class SpaceGenerator:
            @staticmethod
            def create(*args, **kwargs):
                return Mock()
            
        class space_generator:
            class PostOrderApply:
                def __init__(self, *args, **kwargs):
                    pass
                    
        class schedule_rule:
            MultiLevelTiling = Mock
            MultiLevelTilingTensorCore = Mock
            AutoInline = Mock
            CrossThreadReduction = Mock
            AddRFactor = Mock
            ParallelizeVectorizeUnroll = Mock
            AutoBind = Mock
            
        class postproc:
            RewriteCooperativeFetch = Mock
            RewriteParallelVectorizeUnroll = Mock
            RewriteTensorize = Mock
            RewriteUnboundBlock = Mock
            VerifyGPUCode = Mock
            DisallowDynamicLoop = Mock
            RewriteReductionBlock = Mock
            RewriteLayout = Mock
            
        class mutator:
            MutateTileSize = Mock
            MutateUnroll = Mock
            MutateThreadBinding = Mock
            
            @staticmethod
            def create(kind):
                return {Mock(): 0.9, Mock(): 0.08, Mock(): 0.02}
    
    class Target:
        def __init__(self, target_str):
            self.attrs = {}
            if "sm_" in target_str:
                arch = target_str.split("sm_")[1].split()[0]
                self.attrs["arch"] = f"sm_{arch}"
    
    class contrib:
        class nvcc:
            @staticmethod
            def have_tensorcore(target=None):
                return True


class TestSwiftSpaceGenerator(unittest.TestCase):
    """Test cases for SwiftSpaceGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        logging.basicConfig(level=logging.DEBUG)
        
    @patch('swift_space_generator.TVM_AVAILABLE', True)
    @patch('swift_space_generator.tvm', MockTVM)
    @patch('swift_space_generator.ms', MockTVM.meta_schedule)
    def test_swift_space_generator_creation(self):
        """Test basic Swift space generator creation."""
        factory = SwiftSpaceGenerator(target_str="cuda -arch=sm_80")
        
        self.assertEqual(factory.target_str, "cuda -arch=sm_80")
        self.assertTrue(factory.tensor_core_available)
        
    @patch('swift_space_generator.TVM_AVAILABLE', True)
    @patch('swift_space_generator.tvm', MockTVM)
    @patch('swift_space_generator.ms', MockTVM.meta_schedule)
    def test_tensorcore_detection(self):
        """Test TensorCore detection logic."""
        # Test modern architecture
        factory = SwiftSpaceGenerator(target_str="cuda -arch=sm_80")
        self.assertTrue(factory.tensor_core_available)
        
        # Test older architecture
        factory = SwiftSpaceGenerator(target_str="cuda -arch=sm_60")
        self.assertFalse(factory.tensor_core_available)
        
        # Test non-CUDA target
        factory = SwiftSpaceGenerator(target_str="llvm")
        self.assertFalse(factory.tensor_core_available)
    
    @patch('swift_space_generator.TVM_AVAILABLE', True)
    @patch('swift_space_generator.tvm', MockTVM)
    @patch('swift_space_generator.ms', MockTVM.meta_schedule)
    def test_swift_rule_creation(self):
        """Test Swift schedule rule creation."""
        factory = SwiftSpaceGenerator(target_str="cuda -arch=sm_80")
        rules = factory._get_swift_schedule_rules()
        
        # Should create multiple rules
        self.assertGreater(len(rules), 0)
        
    @patch('swift_space_generator.TVM_AVAILABLE', True)
    @patch('swift_space_generator.tvm', MockTVM)
    @patch('swift_space_generator.ms', MockTVM.meta_schedule)
    def test_swift_postprocessor_creation(self):
        """Test Swift postprocessor creation."""
        factory = SwiftSpaceGenerator(target_str="cuda -arch=sm_80")
        postprocs = factory._get_swift_postprocessors()
        
        # Should create multiple postprocessors
        self.assertGreater(len(postprocs), 0)
    
    @patch('swift_space_generator.TVM_AVAILABLE', True)
    @patch('swift_space_generator.tvm', MockTVM)
    @patch('swift_space_generator.ms', MockTVM.meta_schedule)
    def test_swift_mutator_creation(self):
        """Test Swift mutator creation."""
        factory = SwiftSpaceGenerator(target_str="cuda -arch=sm_80")
        mutators = factory._get_swift_mutator_probs()
        
        # Should create mutator dictionary
        self.assertIsInstance(mutators, dict)
        self.assertGreater(len(mutators), 0)
        
    def test_swift_block_filter(self):
        """Test Swift block filter function."""
        # Mock schedule and block
        mock_sch = Mock()
        mock_block_rv = Mock()
        mock_block = Mock()
        mock_block.name_hint = "matmul_compute"
        mock_sch.get.return_value = mock_block
        
        # Should accept compute blocks
        result = swift_block_filter(mock_sch, mock_block_rv)
        self.assertTrue(result)
        
        # Should reject cache blocks
        mock_block.name_hint = "cache_read"
        result = swift_block_filter(mock_sch, mock_block_rv)
        self.assertFalse(result)
        
    def test_without_tvm(self):
        """Test behavior when TVM is not available."""
        with patch('swift_space_generator.TVM_AVAILABLE', False):
            factory = SwiftSpaceGenerator()
            self.assertEqual(factory.capabilities, {})
            self.assertFalse(factory.tensor_core_available)
            
            space = factory.create_space_generator()
            self.assertIsNone(space)


class TestSwiftScheduleRules(unittest.TestCase):
    """Test cases for Swift schedule rules."""
    
    @patch('swift_schedule_rules.TVM_AVAILABLE', True)
    @patch('swift_schedule_rules.tvm', MockTVM)
    @patch('swift_schedule_rules.ms', MockTVM.meta_schedule)
    def test_tensorcore_rule_creation(self):
        """Test TensorCore rule creation."""
        rule = SwiftHighParallelismRule.create_tensorcore_rule()
        self.assertIsNotNone(rule)
        
    @patch('swift_schedule_rules.TVM_AVAILABLE', True)
    @patch('swift_schedule_rules.tvm', MockTVM)
    @patch('swift_schedule_rules.ms', MockTVM.meta_schedule)
    def test_simt_rule_creation(self):
        """Test SIMT rule creation."""
        rule = SwiftHighParallelismRule.create_simt_rule()
        self.assertIsNotNone(rule)
        
    def test_occupancy_estimation(self):
        """Test occupancy estimation calculations."""
        occupancy = SwiftOccupancyRule.estimate_occupancy(
            threads_per_block=256,
            shared_mem_per_block=8192,
            registers_per_thread=32,
            target="cuda -arch=sm_80"
        )
        
        self.assertIsInstance(occupancy, float)
        self.assertGreaterEqual(occupancy, 0.0)
        self.assertLessEqual(occupancy, 1.0)
        
    def test_occupancy_edge_cases(self):
        """Test occupancy estimation edge cases."""
        # Very high resource usage (should limit occupancy)
        low_occ = SwiftOccupancyRule.estimate_occupancy(
            threads_per_block=1024,
            shared_mem_per_block=98304,  # 96KB
            registers_per_thread=255,
            target="cuda -arch=sm_80"
        )
        
        # Low resource usage (should allow high occupancy)
        high_occ = SwiftOccupancyRule.estimate_occupancy(
            threads_per_block=64,
            shared_mem_per_block=1024,   # 1KB
            registers_per_thread=8,
            target="cuda -arch=sm_80"
        )
        
        self.assertLess(low_occ, high_occ)
        
    @patch('swift_schedule_rules.TVM_AVAILABLE', True)
    @patch('swift_schedule_rules.tvm', MockTVM)
    @patch('swift_schedule_rules.ms', MockTVM.meta_schedule)
    def test_complete_rule_set_creation(self):
        """Test complete Swift rule set creation."""
        rules = create_swift_schedule_rules(
            target="cuda -arch=sm_80",
            enable_tensorcore=True
        )
        
        self.assertIsInstance(rules, list)
        self.assertGreater(len(rules), 0)


class TestSwiftPostprocs(unittest.TestCase):
    """Test cases for Swift postprocessors."""
    
    @patch('swift_postprocs.TVM_AVAILABLE', True)
    @patch('swift_postprocs.tvm', MockTVM)
    @patch('swift_postprocs.ms', MockTVM.meta_schedule)
    def test_cooperative_memory_opts(self):
        """Test cooperative memory optimization postprocessors."""
        coop_fetch = SwiftCooperativeMemoryOpts.create_cooperative_fetch()
        self.assertIsNotNone(coop_fetch)
        
        vec_rewriter = SwiftCooperativeMemoryOpts.create_vectorization_rewriter()
        self.assertIsNotNone(vec_rewriter)
        
    @patch('swift_postprocs.TVM_AVAILABLE', True)
    @patch('swift_postprocs.tvm', MockTVM)
    @patch('swift_postprocs.ms', MockTVM.meta_schedule)
    def test_tensorcore_opts(self):
        """Test TensorCore optimization postprocessors."""
        tensorize = SwiftTensorCoreOpts.create_tensorize_rewriter()
        self.assertIsNotNone(tensorize)
        
    @patch('swift_postprocs.TVM_AVAILABLE', True)
    @patch('swift_postprocs.tvm', MockTVM)
    @patch('swift_postprocs.ms', MockTVM.meta_schedule)
    def test_gpu_verification(self):
        """Test GPU verification postprocessors."""
        gpu_verifier = SwiftGPUVerification.create_gpu_verifier()
        self.assertIsNotNone(gpu_verifier)
        
        dynamic_disallower = SwiftGPUVerification.create_dynamic_loop_disallower()
        self.assertIsNotNone(dynamic_disallower)
        
    @patch('swift_postprocs.TVM_AVAILABLE', True)
    @patch('swift_postprocs.tvm', MockTVM)
    @patch('swift_postprocs.ms', MockTVM.meta_schedule)
    def test_postprocessor_sequence(self):
        """Test postprocessor sequence management."""
        sequence = SwiftPostprocessorSequence(
            target="cuda -arch=sm_80",
            enable_tensorcore=True
        )
        
        ordered = sequence.get_ordered_postprocessors()
        self.assertIsInstance(ordered, list)
        
        summary = sequence.get_phase_summary()
        self.assertIsInstance(summary, dict)
        
        # Check phase structure
        expected_phases = ["memory_opt", "compute_opt", "block_opt", "layout_opt", "verification"]
        for phase in expected_phases:
            self.assertIn(phase, summary)
            
    @patch('swift_postprocs.TVM_AVAILABLE', True)
    @patch('swift_postprocs.tvm', MockTVM)
    @patch('swift_postprocs.ms', MockTVM.meta_schedule)
    def test_complete_postproc_creation(self):
        """Test complete Swift postprocessor creation."""
        postprocs = create_swift_postprocessors(
            target="cuda -arch=sm_80",
            enable_tensorcore=True
        )
        
        self.assertIsInstance(postprocs, list)
        self.assertGreater(len(postprocs), 0)


class TestAPIDiscovery(unittest.TestCase):
    """Test cases for API discovery functionality."""
    
    @patch('swift_space_generator.TVM_AVAILABLE', True)
    @patch('swift_space_generator.tvm', MockTVM)
    @patch('swift_space_generator.ms', MockTVM.meta_schedule)
    def test_api_discovery(self):
        """Test API discovery process."""
        capabilities = discover_available_apis()
        
        self.assertIsInstance(capabilities, dict)
        
        # Check expected keys
        expected_keys = [
            "schedule_rules", "postprocs", "mutators", 
            "tensor_core_support", "space_generator_methods"
        ]
        for key in expected_keys:
            self.assertIn(key, capabilities)
            
        # Check that lists are populated
        self.assertIsInstance(capabilities["schedule_rules"], list)
        self.assertIsInstance(capabilities["postprocs"], list)
        self.assertIsInstance(capabilities["mutators"], list)
        
    def test_api_discovery_without_tvm(self):
        """Test API discovery when TVM is not available."""
        with patch('swift_space_generator.TVM_AVAILABLE', False):
            capabilities = discover_available_apis()
            self.assertEqual(capabilities, {})


class TestIntegration(unittest.TestCase):
    """Integration tests for Swift components."""
    
    @patch('swift_space_generator.TVM_AVAILABLE', True)
    @patch('swift_space_generator.tvm', MockTVM)
    @patch('swift_space_generator.ms', MockTVM.meta_schedule)
    @patch('swift_schedule_rules.TVM_AVAILABLE', True)
    @patch('swift_schedule_rules.tvm', MockTVM)
    @patch('swift_schedule_rules.ms', MockTVM.meta_schedule)
    @patch('swift_postprocs.TVM_AVAILABLE', True)
    @patch('swift_postprocs.tvm', MockTVM)
    @patch('swift_postprocs.ms', MockTVM.meta_schedule)
    def test_end_to_end_space_creation(self):
        """Test end-to-end Swift space generator creation."""
        factory = SwiftSpaceGenerator(target_str="cuda -arch=sm_80")
        space = factory.create_space_generator()
        
        # Should successfully create a space generator
        self.assertIsNotNone(space)
        
        # Check that components were created
        self.assertTrue(factory.capabilities)
        self.assertTrue(factory.tensor_core_available)
        
    def test_configuration_consistency(self):
        """Test that configurations are consistent across components."""
        target = "cuda -arch=sm_80"
        
        # Check that all components can be created with same target
        with patch('swift_space_generator.TVM_AVAILABLE', True), \
             patch('swift_space_generator.tvm', MockTVM), \
             patch('swift_space_generator.ms', MockTVM.meta_schedule), \
             patch('swift_schedule_rules.TVM_AVAILABLE', True), \
             patch('swift_schedule_rules.tvm', MockTVM), \
             patch('swift_schedule_rules.ms', MockTVM.meta_schedule), \
             patch('swift_postprocs.TVM_AVAILABLE', True), \
             patch('swift_postprocs.tvm', MockTVM), \
             patch('swift_postprocs.ms', MockTVM.meta_schedule):
            
            factory = SwiftSpaceGenerator(target_str=target)
            rules = create_swift_schedule_rules(target=target)
            postprocs = create_swift_postprocessors(target=target)
            
            # All should be created successfully
            self.assertIsNotNone(factory)
            self.assertIsInstance(rules, list)
            self.assertIsInstance(postprocs, list)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)