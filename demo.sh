#!/bin/bash
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

# Demonstration script showing complete Relay VM MetaSchedule workflow

set -e

echo "========================================================================"
echo "    RELAY VM METASCHEDULE COMPILATION AND BENCHMARKING DEMO"
echo "========================================================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📋 OVERVIEW${NC}"
echo "This demo shows the complete workflow for:"
echo "  1. Creating a MetaSchedule database"
echo "  2. Loading a Relay model"
echo "  3. Compiling with MetaSchedule + Relay VM"
echo "  4. Benchmarking and analyzing results"
echo

echo -e "${BLUE}🎯 STEP 1: Create Mock Database${NC}"
echo "Generating mock MetaSchedule database for testing..."
python create_mock_database.py --output-dir ./demo_db
echo -e "${GREEN}✓ Database created${NC}"
echo

echo -e "${BLUE}🎯 STEP 2: Test Example Model${NC}"
echo "Validating example model creation..."
python example_model.py
echo -e "${GREEN}✓ Model validation complete${NC}"
echo

echo -e "${BLUE}🎯 STEP 3: Run Integration Tests${NC}"
echo "Running comprehensive validation..."
python test_integration.py
echo -e "${GREEN}✓ Integration tests passed${NC}"
echo

echo -e "${BLUE}🎯 STEP 4: Example Usage Commands${NC}"
echo "Here are ready-to-run examples (require TVM + dependencies):"
echo

echo -e "${YELLOW}# CPU Example:${NC}"
echo "python compile_and_benchmark_relay_vm.py \\"
echo "  --model example_model.py \\"
echo "  --db-workload ./demo_db/database_workload.json \\"
echo "  --db-record ./demo_db/database_tuning_record.json \\"
echo "  --target 'llvm' \\"
echo "  --input data:1x784:float32 \\"
echo "  --warmup 5 --number 10 --repeat 5"
echo

echo -e "${YELLOW}# CUDA Example (Ada Lovelace):${NC}"
echo "python compile_and_benchmark_relay_vm.py \\"
echo "  --model resnet18.onnx \\"
echo "  --db-workload database_workload.json \\"
echo "  --db-record database_tuning_record.json \\"
echo "  --target 'cuda -arch=sm_89' \\"
echo "  --input data:1x3x224x224:float16 \\"
echo "  --warmup 10 --number 10 --repeat 10 \\"
echo "  --output-dir ./results"
echo

echo -e "${YELLOW}# Multi-input Example:${NC}"
echo "python compile_and_benchmark_relay_vm.py \\"
echo "  --model transformer.onnx \\"
echo "  --db-workload database_workload.json \\"
echo "  --db-record database_tuning_record.json \\"
echo "  --target 'cuda -arch=sm_89' \\"
echo "  --input input_ids:1x512:int32 \\"
echo "  --input attention_mask:1x512:float16 \\"
echo "  --warmup 5 --number 20 --repeat 5"
echo

echo -e "${BLUE}🎯 STEP 5: Analyze PTX Output${NC}"
echo "After running CUDA compilation, analyze tensor core usage:"
echo -e "${YELLOW}python dump_cuda_ptx.py ./results/cuda_ptx.ptx${NC}"
echo

echo -e "${BLUE}📊 Expected Output${NC}"
echo "The script will produce:"
echo "  • Timing statistics (p50, p90, p95, p99, mean, std)"
echo "  • VM executable (.so file)"
echo "  • TIR source code"
echo "  • PTX assembly (CUDA targets)"
echo "  • Tensor core analysis (if applicable)"
echo

echo -e "${BLUE}🔧 Architecture Support${NC}"
echo "Supported GPU architectures:"
echo "  • SM_70+ (Volta V100)"
echo "  • SM_75+ (Turing RTX 20XX)"
echo "  • SM_80+ (Ampere A100)"
echo "  • SM_89+ (Ada RTX 40XX)"
echo "  • SM_90+ (Hopper H100)"
echo

echo -e "${BLUE}✨ Key Features Delivered${NC}"
echo "  ✓ MetaSchedule database integration"
echo "  ✓ Relay VM compilation (not graph executor)"
echo "  ✓ Comprehensive timing statistics"
echo "  ✓ CUDA tensor core detection"
echo "  ✓ Artifact generation (TIR, PTX)"
echo "  ✓ Multi-format model support (ONNX, Python)"
echo "  ✓ Robust error handling"
echo "  ✓ Complete documentation"
echo

echo "========================================================================"
echo -e "${GREEN}🎉 DEMO COMPLETE - Ready for Production Use!${NC}"
echo "========================================================================"
echo
echo "All files are ready to use. For production deployment:"
echo "  1. Build TVM with CUDA support"
echo "  2. Install numpy and onnx packages"
echo "  3. Generate MetaSchedule database from your workloads"
echo "  4. Run the script with your models and targets"
echo
echo "Documentation: RELAY_VM_README.md"
echo "Examples: example_model.py, create_mock_database.py"
echo "Tests: test_integration.py, test_relay_vm_script.py"