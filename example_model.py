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
Example Python model file for testing Relay VM MetaSchedule script.
Creates a simple MLP model for demonstration.
"""

def get_model():
    """Return a simple MLP model for testing."""
    try:
        import tvm
        from tvm import relay
        import numpy as np
    except ImportError as e:
        raise ImportError(f"TVM is required: {e}")
    
    # Input tensor
    data = relay.var("data", shape=(1, 784), dtype="float32")
    
    # First layer: dense + relu
    weight1 = relay.var("weight1", shape=(128, 784), dtype="float32")
    bias1 = relay.var("bias1", shape=(128,), dtype="float32")
    dense1 = relay.nn.dense(data, weight1)
    dense1 = relay.nn.bias_add(dense1, bias1)
    relu1 = relay.nn.relu(dense1)
    
    # Second layer: dense + relu
    weight2 = relay.var("weight2", shape=(64, 128), dtype="float32")
    bias2 = relay.var("bias2", shape=(64,), dtype="float32")
    dense2 = relay.nn.dense(relu1, weight2)
    dense2 = relay.nn.bias_add(dense2, bias2)
    relu2 = relay.nn.relu(dense2)
    
    # Output layer: dense
    weight3 = relay.var("weight3", shape=(10, 64), dtype="float32")
    bias3 = relay.var("bias3", shape=(10,), dtype="float32")
    dense3 = relay.nn.dense(relu2, weight3)
    output = relay.nn.bias_add(dense3, bias3)
    
    # Create function
    func = relay.Function([data, weight1, bias1, weight2, bias2, weight3, bias3], output)
    mod = tvm.IRModule.from_expr(func)
    
    # Create random parameters
    params = {
        "weight1": tvm.nd.array(np.random.randn(128, 784).astype("float32")),
        "bias1": tvm.nd.array(np.random.randn(128).astype("float32")),
        "weight2": tvm.nd.array(np.random.randn(64, 128).astype("float32")),
        "bias2": tvm.nd.array(np.random.randn(64).astype("float32")),
        "weight3": tvm.nd.array(np.random.randn(10, 64).astype("float32")),
        "bias3": tvm.nd.array(np.random.randn(10).astype("float32"))
    }
    
    return mod, params


def get_conv_model():
    """Return a simple CNN model for testing."""
    try:
        import tvm
        from tvm import relay
        import numpy as np
    except ImportError as e:
        raise ImportError(f"TVM is required: {e}")
    
    # Input: NCHW format
    data = relay.var("data", shape=(1, 3, 32, 32), dtype="float32")
    
    # Conv layer 1
    weight1 = relay.var("weight1", shape=(16, 3, 3, 3), dtype="float32")
    conv1 = relay.nn.conv2d(data, weight1, padding=(1, 1))
    relu1 = relay.nn.relu(conv1)
    pool1 = relay.nn.max_pool2d(relu1, pool_size=(2, 2))
    
    # Conv layer 2  
    weight2 = relay.var("weight2", shape=(32, 16, 3, 3), dtype="float32")
    conv2 = relay.nn.conv2d(pool1, weight2, padding=(1, 1))
    relu2 = relay.nn.relu(conv2)
    pool2 = relay.nn.max_pool2d(relu2, pool_size=(2, 2))
    
    # Flatten and dense
    flat = relay.nn.batch_flatten(pool2)
    weight3 = relay.var("weight3", shape=(10, 2048), dtype="float32")  # 32*8*8 = 2048
    dense = relay.nn.dense(flat, weight3)
    
    func = relay.Function([data, weight1, weight2, weight3], dense)
    mod = tvm.IRModule.from_expr(func)
    
    params = {
        "weight1": tvm.nd.array(np.random.randn(16, 3, 3, 3).astype("float32")),
        "weight2": tvm.nd.array(np.random.randn(32, 16, 3, 3).astype("float32")),
        "weight3": tvm.nd.array(np.random.randn(10, 2048).astype("float32"))
    }
    
    return mod, params


# For backward compatibility
def get_workload():
    """Alias for get_model()."""
    return get_model()


def build_model():
    """Alias for get_model().""" 
    return get_model()


if __name__ == "__main__":
    # Test the model creation
    print("Testing MLP model creation...")
    try:
        mod, params = get_model()
        print(f"✓ MLP model created successfully")
        print(f"  - Module: {type(mod)}")
        print(f"  - Parameters: {len(params)} tensors")
    except Exception as e:
        print(f"✗ MLP model creation failed: {e}")
    
    print("\nTesting CNN model creation...")
    try:
        mod, params = get_conv_model()
        print(f"✓ CNN model created successfully")
        print(f"  - Module: {type(mod)}")
        print(f"  - Parameters: {len(params)} tensors")
    except Exception as e:
        print(f"✗ CNN model creation failed: {e}")