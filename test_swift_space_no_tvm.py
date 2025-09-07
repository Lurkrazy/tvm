#!/usr/bin/env python3
"""
Unit tests for Swift-style Space Generator (TVM-independent)

Tests API discovery, component creation, and basic functionality
without requiring TVM to be installed or built.
"""

import unittest
import logging
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Force TVM_AVAILABLE to False to test fallback behavior
import swift_space_generator
import swift_schedule_rules
import swift_postprocs

# Patch the TVM_AVAILABLE flags
swift_space_generator.TVM_AVAILABLE = False
swift_schedule_rules.TVM_AVAILABLE = False
swift_postprocs.TVM_AVAILABLE = False

from swift_space_generator import SwiftSpaceGenerator, discover_available_apis, swift_block_filter
from swift_schedule_rules import SwiftOccupancyRule, create_swift_schedule_rules
from swift_postprocs import SwiftPostprocessorSequence, create_swift_postprocessors


class TestSwiftSpaceGeneratorWithoutTVM(unittest.TestCase):
    """Test cases for SwiftSpaceGenerator without TVM."""
    
    def setUp(self):
        """Set up test fixtures."""
        logging.basicConfig(level=logging.DEBUG)
        
    def test_swift_space_generator_creation_without_tvm(self):
        """Test Swift space generator creation without TVM."""
        factory = SwiftSpaceGenerator(target_str="cuda -arch=sm_80")
        
        self.assertEqual(factory.target_str, "cuda -arch=sm_80")
        self.assertFalse(factory.tensor_core_available)
        self.assertEqual(factory.capabilities, {})
        
    def test_space_creation_returns_none_without_tvm(self):
        """Test that space creation returns None without TVM."""
        factory = SwiftSpaceGenerator(target_str="cuda -arch=sm_80")
        space = factory.create_space_generator()
        
        self.assertIsNone(space)
        
    def test_api_discovery_without_tvm(self):
        """Test API discovery when TVM is not available."""
        capabilities = discover_available_apis()
        self.assertEqual(capabilities, {})
        
    def test_swift_block_filter_without_dependencies(self):
        """Test Swift block filter with mock objects."""
        # This function should work even without TVM
        # Create mock objects
        class MockBlock:
            def __init__(self, name):
                self.name_hint = name
                
        class MockSchedule:
            def __init__(self, block):
                self.block = block
                
            def get(self, block_rv):
                return self.block
        
        # Test compute block acceptance
        compute_block = MockBlock("matmul_compute")
        mock_sch = MockSchedule(compute_block)
        result = swift_block_filter(mock_sch, None)
        self.assertTrue(result)
        
        # Test cache block rejection
        cache_block = MockBlock("cache_read")
        mock_sch = MockSchedule(cache_block)
        result = swift_block_filter(mock_sch, None)
        self.assertFalse(result)
        
        # Test error handling
        def error_sch(block_rv):
            raise Exception("Test error")
        mock_sch.get = error_sch
        result = swift_block_filter(mock_sch, None)
        self.assertFalse(result)


class TestSwiftScheduleRulesWithoutTVM(unittest.TestCase):
    """Test cases for Swift schedule rules without TVM."""
    
    def test_occupancy_estimation_without_tvm(self):
        """Test occupancy estimation calculations (TVM-independent)."""
        occupancy = SwiftOccupancyRule.estimate_occupancy(
            threads_per_block=256,
            shared_mem_per_block=8192,
            registers_per_thread=32,
            target="cuda -arch=sm_80"
        )
        
        self.assertIsInstance(occupancy, float)
        self.assertGreaterEqual(occupancy, 0.0)
        self.assertLessEqual(occupancy, 1.0)
        
    def test_occupancy_resource_limits(self):
        """Test occupancy with different resource constraints."""
        # Test thread limit
        occ_threads = SwiftOccupancyRule.estimate_occupancy(
            threads_per_block=1024,  # High thread count
            shared_mem_per_block=1024,  # Low shared mem
            registers_per_thread=8,   # Low registers
        )
        
        # Test shared memory limit
        occ_shmem = SwiftOccupancyRule.estimate_occupancy(
            threads_per_block=64,    # Low thread count
            shared_mem_per_block=98304,  # High shared mem (96KB)
            registers_per_thread=8,   # Low registers
        )
        
        # Test register limit
        occ_regs = SwiftOccupancyRule.estimate_occupancy(
            threads_per_block=64,    # Low thread count
            shared_mem_per_block=1024,  # Low shared mem
            registers_per_thread=255, # High registers
        )
        
        # All should be valid occupancy values
        for occ in [occ_threads, occ_shmem, occ_regs]:
            self.assertGreaterEqual(occ, 0.0)
            self.assertLessEqual(occ, 1.0)
    
    def test_occupancy_architecture_detection(self):
        """Test occupancy calculation with different architectures."""
        # Test Ampere architecture (sm_80+)
        occ_ampere = SwiftOccupancyRule.estimate_occupancy(
            threads_per_block=256,
            shared_mem_per_block=50000,  # 50KB
            registers_per_thread=32,
            target="cuda -arch=sm_80"
        )
        
        # Test Volta architecture (sm_70)
        occ_volta = SwiftOccupancyRule.estimate_occupancy(
            threads_per_block=256,
            shared_mem_per_block=50000,  # 50KB
            registers_per_thread=32,
            target="cuda -arch=sm_70"
        )
        
        # Ampere should handle shared memory better (larger limit)
        # This test verifies the architecture detection works
        self.assertIsInstance(occ_ampere, float)
        self.assertIsInstance(occ_volta, float)
        
    def test_schedule_rules_creation_without_tvm(self):
        """Test schedule rules creation returns empty list without TVM."""
        rules = create_swift_schedule_rules()
        self.assertEqual(rules, [])


class TestSwiftPostprocsWithoutTVM(unittest.TestCase):
    """Test cases for Swift postprocessors without TVM."""
    
    def test_postprocessor_sequence_structure(self):
        """Test postprocessor sequence structure without TVM."""
        sequence = SwiftPostprocessorSequence(
            target="cuda -arch=sm_80",
            enable_tensorcore=True
        )
        
        # Check phase structure exists
        self.assertIsInstance(sequence.phases, dict)
        
        expected_phases = ["memory_opt", "compute_opt", "block_opt", "layout_opt", "verification"]
        for phase in expected_phases:
            self.assertIn(phase, sequence.phases)
            
    def test_postprocessor_phase_summary(self):
        """Test postprocessor phase summary generation."""
        sequence = SwiftPostprocessorSequence(target="cuda")
        summary = sequence.get_phase_summary()
        
        self.assertIsInstance(summary, dict)
        
        # All phases should have 0 postprocessors without TVM
        for phase, count in summary.items():
            self.assertEqual(count, 0)
            
    def test_postprocessor_creation_without_tvm(self):
        """Test postprocessor creation returns empty list without TVM."""
        postprocs = create_swift_postprocessors()
        self.assertEqual(postprocs, [])


class TestErrorHandling(unittest.TestCase):
    """Test error handling and graceful degradation."""
    
    def test_block_filter_error_handling(self):
        """Test block filter handles errors gracefully."""
        # Mock schedule that throws exception
        class ErrorSchedule:
            def get(self, block_rv):
                raise RuntimeError("Mock error")
        
        result = swift_block_filter(ErrorSchedule(), None)
        self.assertFalse(result)  # Should return False on error
        
    def test_occupancy_edge_cases(self):
        """Test occupancy calculation with edge cases."""
        # Zero values (should not crash)
        occ_zero_threads = SwiftOccupancyRule.estimate_occupancy(
            threads_per_block=0,
            shared_mem_per_block=1024,
            registers_per_thread=32
        )
        
        occ_zero_shmem = SwiftOccupancyRule.estimate_occupancy(
            threads_per_block=256,
            shared_mem_per_block=0,
            registers_per_thread=32
        )
        
        occ_zero_regs = SwiftOccupancyRule.estimate_occupancy(
            threads_per_block=256,
            shared_mem_per_block=1024,
            registers_per_thread=0
        )
        
        # Should handle gracefully (not crash)
        for occ in [occ_zero_threads, occ_zero_shmem, occ_zero_regs]:
            self.assertIsInstance(occ, float)
            self.assertGreaterEqual(occ, 0.0)
            
    def test_invalid_architecture_parsing(self):
        """Test handling of invalid architecture strings."""
        # Should not crash with invalid target
        occ = SwiftOccupancyRule.estimate_occupancy(
            threads_per_block=256,
            shared_mem_per_block=1024,
            registers_per_thread=32,
            target="invalid-target"
        )
        
        self.assertIsInstance(occ, float)
        

class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation and consistency."""
    
    def test_tensorcore_detection_logic(self):
        """Test TensorCore detection logic."""
        factory = SwiftSpaceGenerator(target_str="cuda -arch=sm_80")
        
        # Without TVM, should default to False
        self.assertFalse(factory.tensor_core_available)
        
    def test_target_string_handling(self):
        """Test target string parsing and handling."""
        targets = [
            "cuda",
            "cuda -arch=sm_70",
            "cuda -arch=sm_80",
            "llvm",
            "invalid-target"
        ]
        
        for target in targets:
            factory = SwiftSpaceGenerator(target_str=target)
            # Should not crash during creation
            self.assertEqual(factory.target_str, target)
            
    def test_component_consistency(self):
        """Test that components are created consistently."""
        target = "cuda -arch=sm_80"
        
        # All components should handle the same target
        factory = SwiftSpaceGenerator(target_str=target)
        rules = create_swift_schedule_rules(target=target)
        postprocs = create_swift_postprocessors(target=target)
        
        # Should all return empty/None without TVM, but not crash
        self.assertIsNone(factory.create_space_generator())
        self.assertEqual(rules, [])
        self.assertEqual(postprocs, [])


class TestDocumentationExamples(unittest.TestCase):
    """Test examples from documentation work correctly."""
    
    def test_basic_usage_example(self):
        """Test basic usage example from README."""
        # This should not crash even without TVM
        factory = SwiftSpaceGenerator(target_str="cuda -arch=sm_80")
        space = factory.create_space_generator()
        
        # Should return None gracefully
        self.assertIsNone(space)
        
    def test_occupancy_example(self):
        """Test occupancy estimation example."""
        occupancy = SwiftOccupancyRule.estimate_occupancy(
            threads_per_block=256,
            shared_mem_per_block=8192,  # 8KB
            registers_per_thread=32,
            target="cuda -arch=sm_80"
        )
        
        # Should return valid occupancy
        self.assertIsInstance(occupancy, float)
        self.assertGreater(occupancy, 0.0)
        self.assertLessEqual(occupancy, 1.0)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Running Swift Space Generator Tests (TVM-independent)")
    print("=" * 60)
    
    # Run tests
    unittest.main(verbosity=2)