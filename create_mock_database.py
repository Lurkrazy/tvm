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
Create mock MetaSchedule database files for testing.
"""

import json
import os
from pathlib import Path
import argparse


def create_mock_database(output_dir: str = "./mock_db"):
    """Create mock MetaSchedule database files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Mock workload data
    workloads = [
        {
            "workload_key": ["test_workload_dense"],
            "mod": {
                "__tvm_object__": "IRModule", 
                "__tvm_version__": "0.13.0",
                "functions": {
                    "test_func": {
                        "__tvm_object__": "PrimFunc",
                        "body": {
                            "__tvm_object__": "Block",
                            "name": "root"
                        }
                    }
                }
            },
            "target": {
                "__tvm_object__": "Target",
                "kind": "llvm",
                "keys": ["cpu"]
            }
        },
        {
            "workload_key": ["test_workload_conv2d"],
            "mod": {
                "__tvm_object__": "IRModule",
                "__tvm_version__": "0.13.0", 
                "functions": {
                    "conv2d_func": {
                        "__tvm_object__": "PrimFunc",
                        "body": {
                            "__tvm_object__": "Block",
                            "name": "conv2d_root"
                        }
                    }
                }
            },
            "target": {
                "__tvm_object__": "Target",
                "kind": "cuda",
                "keys": ["cuda", "gpu"]
            }
        }
    ]
    
    # Mock tuning records
    records = [
        {
            "workload_key": ["test_workload_dense"],
            "trace": {
                "__tvm_object__": "Trace",
                "insts": [
                    {
                        "__tvm_object__": "Instruction",
                        "kind": "Sample",
                        "inputs": [],
                        "attrs": []
                    }
                ]
            },
            "run_secs": [0.001234, 0.001187, 0.001256, 0.001203, 0.001189],
            "target": {
                "__tvm_object__": "Target", 
                "kind": "llvm",
                "keys": ["cpu"]
            },
            "args_info": [
                {
                    "__tvm_object__": "ArgInfo",
                    "name": "data",
                    "shape": [1, 784],
                    "dtype": "float32"
                }
            ]
        },
        {
            "workload_key": ["test_workload_conv2d"],
            "trace": {
                "__tvm_object__": "Trace",
                "insts": [
                    {
                        "__tvm_object__": "Instruction",
                        "kind": "BlockRV",
                        "inputs": [],
                        "attrs": ["tensorcore_schedule"]
                    }
                ]
            },
            "run_secs": [0.000876, 0.000891, 0.000863, 0.000888, 0.000879],
            "target": {
                "__tvm_object__": "Target",
                "kind": "cuda", 
                "keys": ["cuda", "gpu"],
                "attrs": {"arch": "sm_80"}
            },
            "args_info": [
                {
                    "__tvm_object__": "ArgInfo",
                    "name": "data",
                    "shape": [1, 3, 32, 32], 
                    "dtype": "float32"
                }
            ]
        }
    ]
    
    # Save workload file
    workload_file = output_path / "database_workload.json"
    with open(workload_file, 'w') as f:
        json.dump(workloads, f, indent=2)
    
    # Save record file
    record_file = output_path / "database_tuning_record.json"
    with open(record_file, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"Mock database created in {output_dir}:")
    print(f"  - Workloads: {workload_file} ({len(workloads)} entries)")
    print(f"  - Records: {record_file} ({len(records)} entries)")
    
    return str(workload_file), str(record_file)


def main():
    """Create mock database files."""
    parser = argparse.ArgumentParser(description="Create mock MetaSchedule database for testing")
    parser.add_argument("--output-dir", default="./mock_db", help="Output directory (default: ./mock_db)")
    
    args = parser.parse_args()
    
    workload_file, record_file = create_mock_database(args.output_dir)
    
    print("\nExample usage:")
    print(f"python compile_and_benchmark_relay_vm.py \\")
    print(f"  --model example_model.py \\")
    print(f"  --db-workload {workload_file} \\")
    print(f"  --db-record {record_file} \\")
    print(f"  --target 'llvm' \\")
    print(f"  --input data:1x784:float32")


if __name__ == "__main__":
    main()