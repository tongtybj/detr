 # Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a
 # copy of this software and associated documentation files (the "Software"),
 # to deal in the Software without restriction, including without limitation
 # the rights to use, copy, modify, merge, publish, distribute, sublicense,
 # and/or sell copies of the Software, and to permit persons to whom the
 # Software is furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 # DEALINGS IN THE SOFTWARE.

from __future__ import print_function
import tensorrt as trt
from onnx.backend.base import Backend, BackendRep, Device, DeviceType, namedtupledict
import onnx
from onnx import helper as onnx_helper
from onnx import numpy_helper
import numpy as np
import six

import sys
sys.path.append('/usr/src/tensorrt/samples/python/')
import common

# HACK Should look for a better way/place to do this
from ctypes import cdll, c_char_p
libcudart = cdll.LoadLibrary('libcudart.so')
libcudart.cudaGetErrorString.restype = c_char_p
def cudaSetDevice(device_idx):
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        raise RuntimeError("cudaSetDevice: " + error_string)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

import tensorrt as trt
import numpy as np

class TensorRTBackendRep(BackendRep):
    def __init__(self, trt_engine_path, device,
            max_workspace_size=None, serialize_engine=False, verbose=False, **kwargs):
        if not isinstance(device, Device):
            device = Device(device)
        self._set_device(device)
        self._logger = TRT_LOGGER
        self.shape_tensor_inputs = []
        self.serialize_engine = serialize_engine
        self.verbose = verbose
        self.dynamic = False

        if self.verbose:
            print(f'\nRunning {model.graph.name}...')
            TRT_LOGGER.min_severity = trt.Logger.VERBOSE

        if not trt.init_libnvinfer_plugins(TRT_LOGGER, ""):
            msg = "Failed to initialize TensorRT's plugin library."
            raise RuntimeError(msg)

        # create tensorrt engine directly from the serialized engine file
        with open(trt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

    def _set_device(self, device):
        self.device = device
        assert(device.type == DeviceType.CUDA)
        cudaSetDevice(device.device_id)

    def run(self, inputs, **kwargs):
        """Execute the prepared engine and return the outputs as a named tuple.
        inputs -- Input tensor(s) as a Numpy array or list of Numpy arrays.
        """
        for i in range(len(inputs)):
            self.inputs[i].host = inputs[i]
        outputs= common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)

        # hard-coding
        if len(outputs) == 1:
            return outputs[0].reshape([256, 1, 256])
        if len(outputs) == 3:
            return outputs[0].reshape([1, 1, 1024, 1]), outputs[1].reshape([1, 1, 1024, 2]), outputs[2].reshape([1, 1, 1024, 2])

        raise ValueError("should not be here!!")

class TensorRTBackend(Backend):
    @classmethod
    def prepare(cls, trt_engine_path, device='CUDA:0', **kwargs):
        """Build an engine from the given model.
        """
        return TensorRTBackendRep(trt_engine_path, device, **kwargs)

prepare         = TensorRTBackend.prepare
