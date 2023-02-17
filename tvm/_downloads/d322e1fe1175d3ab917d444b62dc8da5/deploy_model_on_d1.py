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
.. _tutorial-deploy-model-on-D1:

Deploy the test Model on D1
===========================

This is an example of using Relay to deploy mobilenet on Raspberry Pi.
"""

import tvm
import tvm.relay as relay
from tvm import rpc
import tvm.relay.testing
from tvm.contrib import utils, graph_executor as runtime
from tvm.relay.op.contrib import shl

######################################################################
# Build TVM Runtime on Device
# ---------------------------
#
# The first step is to build the TVM runtime on the remote device.
#
# .. note::
#
#   All instructions in both this section and next section should be
#   executed on the target device, e.g. Raspberry Pi. And we assume it
#   has Linux running.
#
# Since we do compilation on local machine, the remote device is only used
# for running the generated code. We only need to build tvm runtime on
# the remote device.
#
# .. code-block:: bash
#
#   mkdir build
#   cp cmake/config.cmake build
#   cd build
#   cmake ..
#   make runtime -j4 tvm_rpc
#
# After building runtime successfully, we need to copy tvm_rpc and libs
# which used on D1
#

######################################################################
# Set Up RPC Server on Device
# ---------------------------
# To start an RPC server, run the following command on your remote device
#
#   .. code-block:: bash
#
#     ./tvm_rpc server --host=172.16.202.11 --port=9090
#
# If you see the line below, it means the RPC server started
# successfully on your device.
#
#    .. code-block:: bash
#
#      rpc_server.cc:130: bind to 172.16.202.11:9090
#

#################################################################
# Define a Network
# ----------------
# First, we need to define the network with relay frontend API.
# We can load some pre-defined network from :code:`tvm.relay.testing`.
# We can also load models from MXNet, ONNX, PyTorch, and TensorFlow
# (see :ref:`front end tutorials<tutorial-frontend>`).
#
# For convolutional neural networks, although auto-scheduler can work correctly
# with any layout, we found the best performance is typically achieved with NHWC layout.
# We also implemented more optimizations for NHWC layout with the auto-scheduler.
# So it is recommended to convert your models to NHWC layout to use the auto-scheduler.
# You can use :ref:`ConvertLayout <convert-layout-usage>` pass to do the layout conversion in TVM.


def get_network(name, batch_size, layout="NHWC", dtype="float32", use_sparse=False):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet50_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    elif name == "mlp":
        mod, params = relay.testing.mlp.get_workload(
            batch_size=batch_size, dtype=dtype, image_shape=image_shape, num_classes=1000
        )
    else:
        raise ValueError("Network not found.")

    if use_sparse:
        from tvm.topi.sparse.utils import convert_model_dense_to_sparse

        mod, params = convert_model_dense_to_sparse(mod, params, random_params=True)

    return mod, params, input_shape, output_shape


import numpy as np

######################################################################
# Compile The Graph
# -----------------
# To compile the graph, we call the :py:func:`relay.build` function
# with the graph configuration and parameters. However, You cannot to
# deploy a x86 program on a device with RISC-V instruction set. It means
# Relay also needs to know the compilation option of target device,
# apart from arguments :code:`net` and :code:`params` to specify the
# deep learning workload. Actually, the option matters, different option
# will lead to very different performance.

######################################################################
# If we run the example on our x86 server for demonstration, we can simply
# set it as :code:`llvm`. If running it on the Raspberry Pi, we need to
# specify its instruction set. Set :code:`local_demo` to False if you want
# to run this tutorial with a real device.

local_demo = False

if local_demo:
    target = tvm.target.Target("llvm")
else:
    target = tvm.target.Target(
        "llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=sifive-u74 -mabi=lp64d"
    )

network = "mobilenet"
use_sparse = False
batch_size = 1
layout = "NCHW"
dtype = "float32"
log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
print("Get model...")
mod, params, input_shape, output_shape = get_network(
    network, batch_size, layout, dtype=dtype, use_sparse=use_sparse
)

with tvm.transform.PassContext(opt_level=3):
    mod = shl.partition_for_shl(mod, params)

    lib = relay.build(mod, target, params=params)

# After `relay.build`, you will get three return values: graph,
# library and the new parameter, since we do some optimization that will
# change the parameters but keep the result of model as the same.

# Save the library at local temporary directory.
tmp = utils.tempdir()
lib_fname = tmp.relpath("net.so")
if local_demo:
    lib.export_library(lib_fname)
else:
    lib.export_library(lib_fname, cc="riscv64-unknown-linux-gnu-g++")

######################################################################
# Deploy the Model Remotely by RPC
# --------------------------------
# With RPC, you can deploy the model remotely from your host machine
# to the remote device.

# obtain an RPC session from remote device.
if local_demo:
    remote = rpc.LocalSession()
else:
    # The following is my environment, change this to the IP address of your target device
    host = "127.0.0.1"
    port = 9090
    remote = rpc.connect(host, port)

# upload the library to remote device and load it
remote.upload(lib_fname)
rlib = remote.load_module("net.so")

# create the remote runtime module
dev = remote.cpu(0)
module = runtime.GraphModule(rlib["default"](dev))
# set input data
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("data", data_tvm)
# # run
module.run()
# get output
out = module.get_output(0)
# get top1 result
top1 = np.argmax(out.numpy())
print("TVM prediction top-1: {}".format(top1))

# print("Evaluate inference time cost...")
# print(module.benchmark(dev, repeat=1, number=1, min_repeat_ms=500))
