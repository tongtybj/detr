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

def count_trailing_ones(vals):
    count = 0
    for val in reversed(vals):
        if val != 1:
            return count
        count += 1
    return count

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

import tensorrt as trt
import pycuda.driver
import pycuda.gpuarray
import pycuda.autoinit
import numpy as np
from six import string_types

class Binding(object):
    def __init__(self, engine, idx_or_name):
        if isinstance(idx_or_name, string_types):
            self.name = idx_or_name
            self.index  = engine.get_binding_index(self.name)
            if self.index == -1:
                raise IndexError("Binding name not found: %s" % self.name)
        else:
            self.index = idx_or_name
            self.name  = engine.get_binding_name(self.index)
            if self.name is None:
                raise IndexError("Binding index out of range: %i" % self.index)
        self.is_input = engine.binding_is_input(self.index)


        dtype = engine.get_binding_dtype(self.index)
        dtype_map = {trt.DataType.FLOAT: np.float32,
                        trt.DataType.HALF:  np.float16,
                        trt.DataType.INT8:  np.int8,
                        trt.DataType.BOOL: np.bool}
        if hasattr(trt.DataType, 'INT32'):
            dtype_map[trt.DataType.INT32] = np.int32

        self.dtype = dtype_map[dtype]
        shape = engine.get_binding_shape(self.index)

        self.shape = tuple(shape)
        # Must allocate a buffer of size 1 for empty inputs / outputs
        if 0 in self.shape:
            self.empty = True
            # Save original shape to reshape output binding when execution is done
            self.empty_shape = self.shape
            self.shape = tuple([1])
        else:
            self.empty = False
        self._host_buf   = None
        self._device_buf = None
    @property
    def host_buffer(self):
        if self._host_buf is None:
            self._host_buf = pycuda.driver.pagelocked_empty(self.shape, self.dtype)
        return self._host_buf
    @property
    def device_buffer(self):
        if self._device_buf is None:
            self._device_buf = pycuda.gpuarray.empty(self.shape, self.dtype)
        return self._device_buf
    def get_async(self, stream):
        src = self.device_buffer
        dst = self.host_buffer
        src.get_async(stream, dst)
        return dst

def squeeze_hw(x):
    if x.shape[-2:] == (1, 1):
        x = x.reshape(x.shape[:-2])
    elif x.shape[-1] == 1:
        x = x.reshape(x.shape[:-1])
    return x

def check_input_validity(input_idx, input_array, input_binding):
    # Check shape
    trt_shape = tuple(input_binding.shape)
    onnx_shape    = tuple(input_array.shape)

    if onnx_shape != trt_shape:
        if not (trt_shape == (1,) and onnx_shape == ()) :
            raise ValueError("Wrong shape for input %i. Expected %s, got %s." %
                            (input_idx, trt_shape, onnx_shape))

    # Check dtype
    if input_array.dtype != input_binding.dtype:
        #TRT does not support INT64, need to convert to INT32
        if input_array.dtype == np.int64 and input_binding.dtype == np.int32:
            casted_input_array = np.array(input_array, copy=True, dtype=np.int32)
            if np.equal(input_array, casted_input_array).all():
                input_array = casted_input_array
            else:
                raise TypeError("Wrong dtype for input %i. Expected %s, got %s. Cannot safely cast." %
                            (input_idx, input_binding.dtype, input_array.dtype))
        else:
            raise TypeError("Wrong dtype for input %i. Expected %s, got %s." %
                            (input_idx, input_binding.dtype, input_array.dtype))
    return input_array


class Engine(object):
    def __init__(self, trt_engine):
        self.engine = trt_engine
        nbinding = self.engine.num_bindings

        bindings = [Binding(self.engine, i)
                    for i in range(nbinding)]
        self.binding_addrs = [b.device_buffer.ptr for b in bindings]
        self.inputs  = [b for b in bindings if     b.is_input]
        self.outputs = [b for b in bindings if not b.is_input]


        for binding in self.inputs + self.outputs:
            _ = binding.device_buffer # Force buffer allocation
        for binding in self.outputs:
            _ = binding.host_buffer   # Force buffer allocation
        self.context = self.engine.create_execution_context()
        self.stream = pycuda.driver.Stream()

    def __del__(self):
        if self.engine is not None:
            del self.engine

    def run(self, inputs):
        if len(inputs) < len(self.inputs):
            raise ValueError("Not enough inputs. Expected %i, got %i." %
                             (len(self.inputs), len(inputs)))
        if isinstance(inputs, dict):
            inputs = [inputs[b.name] for b in self.inputs]

        for i, (input_array, input_binding) in enumerate(zip(inputs, self.inputs)):
            input_array = check_input_validity(i, input_array, input_binding)
            input_binding_array = input_binding.device_buffer
            input_binding_array.set_async(input_array, self.stream)

        self.context.execute_async_v2(self.binding_addrs, self.stream.handle)

        results = [output.get_async(self.stream)
                   for output in self.outputs]

        # For any empty bindings, update the result shape to the expected empty shape
        for i, (output_array, output_binding) in enumerate(zip(results, self.outputs)):
            if output_binding.empty:
                results[i] = np.empty(shape=output_binding.empty_shape, dtype=output_binding.dtype)

        self.stream.synchronize()
        return results

    def run_no_dma(self, batch_size):
        self.context.execute_async(
            batch_size, self.binding_addrs, self.stream.handle)


class TensorRTBackendRep(BackendRep):
    def __init__(self, model, trt_engine_path, device,
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

        if not isinstance(model, six.string_types):
            model_str = model.SerializeToString()
        else:
            model_str = model

        if not trt.init_libnvinfer_plugins(TRT_LOGGER, ""):
            msg = "Failed to initialize TensorRT's plugin library."
            raise RuntimeError(msg)

        # create tensorrt engine directly from the serialized engine file
        with open(trt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        #self.engine = Engine(trt_engine)

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

        self._output_shapes = {}
        self._output_dtype = {}
        for output in model.graph.output:
            dims = output.type.tensor_type.shape.dim
            output_shape = tuple([dim.dim_value for dim in dims])
            self._output_shapes[output.name] = output_shape
            self._output_dtype[output.name] = output.type.tensor_type.elem_type


    def _set_device(self, device):
        self.device = device
        assert(device.type == DeviceType.CUDA)
        cudaSetDevice(device.device_id)

    def _serialize_deserialize(self, trt_engine):
        return trt_engine

    def run(self, inputs, **kwargs):
        """Execute the prepared engine and return the outputs as a named tuple.
        inputs -- Input tensor(s) as a Numpy array or list of Numpy arrays.
        """
        for i in range(len(inputs)):
            self.inputs[i].host = inputs[i]
        outputs= common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        #np.set_printoptions(threshold=np.inf)
        #print(len(outputs), outputs[0].reshape([32,32]))

        # hard-coding
        return outputs[0].reshape([1, 1, 1024, 1]), outputs[1].reshape([1, 1, 1024, 2]), outputs[2].reshape([1, 1, 1024, 2])

class TensorRTBackend(Backend):
    @classmethod
    def prepare(cls, model, trt_engine_path, device='CUDA:0', **kwargs):
        """Build an engine from the given model.
        model -- An ONNX model as a deserialized protobuf, or a string or file-
                 object containing a serialized protobuf.
        """
        super(TensorRTBackend, cls).prepare(model, device, **kwargs)
        return TensorRTBackendRep(model, trt_engine_path, device, **kwargs)

prepare         = TensorRTBackend.prepare
