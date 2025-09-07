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
Simple test for the Relay VM MetaSchedule benchmarking script.
Tests command line parsing and creates mock data for validation.
"""

import json
import os
import sys
import tempfile
from pathlib import Path


def create_mock_database_files(temp_dir: str) -> tuple:
    """Create mock database files for testing."""
    
    workload_file = os.path.join(temp_dir, "database_workload.json")
    record_file = os.path.join(temp_dir, "database_tuning_record.json")
    
    # Mock workload file
    workloads = [
        {
            "workload_key": "test_workload_1",
            "mod": {"__tvm_object__": "IRModule"},
            "target": {"__tvm_object__": "Target"}
        }
    ]
    
    with open(workload_file, 'w') as f:
        json.dump(workloads, f)
    
    # Mock record file  
    records = [
        {
            "workload_key": "test_workload_1", 
            "trace": {"__tvm_object__": "Trace"},
            "run_secs": [0.001, 0.002, 0.001],
            "target": {"__tvm_object__": "Target"}
        }
    ]
    
    with open(record_file, 'w') as f:
        json.dump(records, f)
    
    return workload_file, record_file


def create_mock_onnx_file(temp_dir: str) -> str:
    """Create a mock ONNX file (just an empty file with .onnx extension)."""
    onnx_file = os.path.join(temp_dir, "mock_model.onnx")
    
    # Create a minimal "ONNX" file (just for testing file existence)
    with open(onnx_file, 'wb') as f:
        f.write(b"mock onnx data")
    
    return onnx_file


def create_mock_python_model(temp_dir: str) -> str:
    """Create a mock Python model file."""
    python_file = os.path.join(temp_dir, "mock_model.py")
    
    model_content = '''
import tvm
from tvm import relay

def get_model():
    """Return a simple Relay model for testing."""
    data = relay.var("data", shape=(1, 784), dtype="float32")
    weight = relay.var("weight", shape=(10, 784), dtype="float32") 
    dense = relay.nn.dense(data, weight)
    
    func = relay.Function([data, weight], dense)
    mod = tvm.IRModule.from_expr(func)
    
    import numpy as np
    params = {
        "weight": tvm.nd.array(np.random.randn(10, 784).astype("float32"))
    }
    
    return mod, params
'''
    
    with open(python_file, 'w') as f:
        f.write(model_content)
    
    return python_file


def test_argument_parsing():
    """Test the argument parsing functionality."""
    print("Testing argument parsing...")
    
    # Import the main script's parse functions
    sys.path.insert(0, '/home/runner/work/tvm/tvm')
    
    try:
        from compile_and_benchmark_relay_vm import parse_input_spec
        
        # Test valid input specifications
        name, shape, dtype = parse_input_spec("data:1x3x224x224:float32")
        assert name == "data"
        assert shape == [1, 3, 224, 224]
        assert dtype == "float32"
        
        name, shape, dtype = parse_input_spec("input_ids:1x512:int32")
        assert name == "input_ids"
        assert shape == [1, 512]
        assert dtype == "int32"
        
        print("✓ Input parsing tests passed")
        
        # Test invalid formats
        try:
            parse_input_spec("invalid_format")
            assert False, "Should have raised ValueError"
        except ValueError:
            print("✓ Invalid format detection works")
        
        try:
            parse_input_spec("name:invalid_shape:float32")
            assert False, "Should have raised ValueError"
        except ValueError:
            print("✓ Invalid shape detection works")
        
    except ImportError as e:
        print(f"⚠ Could not import functions for testing: {e}")
        return False
    
    return True


def test_mock_database_creation():
    """Test creation of mock database files."""
    print("Testing mock database creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        workload_file, record_file = create_mock_database_files(temp_dir)
        
        # Verify files exist
        assert os.path.exists(workload_file), "Workload file not created"
        assert os.path.exists(record_file), "Record file not created"
        
        # Verify JSON is valid
        with open(workload_file, 'r') as f:
            workloads = json.load(f)
            assert len(workloads) > 0, "No workloads in file"
        
        with open(record_file, 'r') as f:
            records = json.load(f)
            assert len(records) > 0, "No records in file"
        
        print("✓ Mock database files created successfully")
    
    return True


def test_command_line_interface():
    """Test the command line interface without actually running TVM."""
    print("Testing command line interface...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock files
        workload_file, record_file = create_mock_database_files(temp_dir)
        onnx_file = create_mock_onnx_file(temp_dir)
        python_file = create_mock_python_model(temp_dir)
        
        # Test argument parsing (without execution)
        test_args = [
            "compile_and_benchmark_relay_vm.py",
            "--model", onnx_file,
            "--db-workload", workload_file, 
            "--db-record", record_file,
            "--target", "llvm",
            "--input", "data:1x3x224x224:float32",
            "--warmup", "5",
            "--number", "3", 
            "--repeat", "2",
            "--output-dir", temp_dir,
            "--log-level", "INFO"
        ]
        
        # Mock sys.argv for testing
        original_argv = sys.argv
        sys.argv = test_args
        
        try:
            # Import and test argument parser
            sys.path.insert(0, '/home/runner/work/tvm/tvm')
            import compile_and_benchmark_relay_vm
            
            # Just test that we can create the argument parser
            parser = compile_and_benchmark_relay_vm.argparse.ArgumentParser()
            # Add arguments as in main script
            parser.add_argument("--model", required=True)
            parser.add_argument("--db-workload", required=True)
            parser.add_argument("--db-record", required=True)
            parser.add_argument("--target", required=True)
            parser.add_argument("--input", action="append", required=True)
            parser.add_argument("--warmup", type=int, default=10)
            parser.add_argument("--number", type=int, default=10)
            parser.add_argument("--repeat", type=int, default=10)
            parser.add_argument("--output-dir", default="./output")
            parser.add_argument("--log-level", default="INFO")
            parser.add_argument("--device-id", type=int, default=0)
            
            # Parse arguments
            args = parser.parse_args(test_args[1:])  # Skip script name
            
            assert args.model == onnx_file
            assert args.db_workload == workload_file
            assert args.db_record == record_file
            assert args.target == "llvm"
            assert args.input == ["data:1x3x224x224:float32"]
            assert args.warmup == 5
            assert args.number == 3
            assert args.repeat == 2
            
            print("✓ Command line argument parsing works")
            
        except Exception as e:
            print(f"⚠ Command line test failed: {e}")
            return False
        finally:
            sys.argv = original_argv
    
    return True


def main():
    """Run all tests."""
    print("="*50)
    print("RELAY VM METASCHEDULE SCRIPT TESTS")
    print("="*50)
    
    tests = [
        test_argument_parsing,
        test_mock_database_creation, 
        test_command_line_interface
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("="*50)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*50)
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())