# -*- coding: utf-8 -*-
import os

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import cv2
import numpy as np

import time
from utils.post_process import *

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return 'Host:\n ' + str(self.host) + '\nDevice:\n' + str(self.device)

    def __repr__(self):
        return self.__str__()

class Detector(object):

    def __init__(self, filepath, img_shape):
        self.engine = self.get_engine(filepath)
        self.allocate_buffs(self.engine)
        self.height, self.width = img_shape
        self.context = self.engine.create_execution_context()

    def get_engine(self, filepath):
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        print("Reading engine from file {}".format(filepath))
        with open(filepath, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffs(self, engine):
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        for binding in engine:
            print(binding)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(binding)), dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

    def inference(self, data):
        data = self.pre_process(data)
        self.inputs[0].host = data.ravel()
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        self.stream.synchronize()
        output = [out.host for out in self.outputs]
        dets = self.post_process(output)

        return dets

    def pre_process(self, data):
        data = data.astype(np.float32) / 255.
        data = data.transpose(2, 0, 1)
        
        return data

    def post_process(self, outputs):
        pred_points = get_predicted_points(outputs[0])
        if len(pred_points) == 0:
            return [], []
        points = list(list(zip(*pred_points))[1])
        slots = inference_slots(points)

        return pred_points, slots
