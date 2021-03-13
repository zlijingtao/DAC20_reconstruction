import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np

def get_centroid(input, grain_size, num_bits, M2D):
    if len(input.size()) == 2:
        print(input.size())
        print(grain_size)
        original_size = input.size()
        reshaped_input = input.view(1, 1, original_size[0], original_size[1])
        pooling_result = F.avg_pool2d(reshaped_input, grain_size, grain_size)
        pooling_result = get_quantized(pooling_result, num_bits, M2D)
        pooling_result = pooling_result.view(pooling_result.size()[2:])
        print(pooling_result.size())
        pooling_result = pooling_result.unsqueeze(1).repeat(1,grain_size[0], 1).view(-1,pooling_result.size()[1]).transpose(0,1)
        output = pooling_result.repeat(1, grain_size[1]).view(-1,pooling_result.size()[1]).transpose(0,1)
    if len(input.size()) == 4:
        original_size = input.size()
        reshaped_input = input.permute(1, 2, 3, 0).view(1, 1, -1, original_size[0])
        pooling_result = F.avg_pool2d(reshaped_input, grain_size, grain_size)
        pooling_result = get_quantized(pooling_result, num_bits, M2D)
        pooling_result = pooling_result.view(pooling_result.size()[2:])
        pooling_result = pooling_result.unsqueeze(1).repeat(1,grain_size[0], 1).view(-1,pooling_result.size()[1]).transpose(0,1)
        pooling_result = pooling_result.repeat(1, grain_size[1]).view(-1,pooling_result.size()[1]).transpose(0,1)
        output = pooling_result.view(original_size[1], original_size[2], original_size[3], original_size[0]).permute(3, 0, 1, 2)
    return output
    
def get_quantized(input, num_bits, M2D):
    output = input.clone()
    if M2D != 0.0:
        qmin = 0
        qmax = qmin + 2.**num_bits - 1.
        scale = 2 * M2D / (qmax - qmin)
        output.div_(scale)
        output.add_((qmax - qmin)/2)
        output.clamp_(qmin, qmax).round_()
        output.add_(-(qmax - qmin)/2)
        output.mul_(scale)
    else:
        output = input.clone().zero_()
    return output

def get_clipped(input, range):
    output = input.clone()
    output.clamp_(-range, range)
    return output

class Unite(torch.autograd.Function):
    def __init__(self, grain_size, num_bits, M2D, save_path):
        super(Unite,self).__init__()
        self.grain_size = grain_size #grain size in tuple
        self.M2D = M2D
        self.num_bits = num_bits
        self.save_path = save_path
    def forward(self, input):
        self.save_for_backward(input)
        self.centroid = get_centroid(input, self.grain_size, self.num_bits, self.M2D)
        global ti
        global num_res
        ti += 1
        input_d = (input - self.centroid)
        output = input.clone().zero_()
        self.W = 1-self.M2D
        output = get_clipped(input_d, self.W)
        if ti <=num_res:
            torch.save(self.centroid, self.save_path + '/saved_tensors/centroid{}.pt'.format(ti))
            torch.save(output, self.save_path + '/saved_tensors/deviation{}.pt'.format(ti))
        output = output + self.centroid

        return output

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class UniteLinear(nn.Linear):

    def __init__(self, infeatures, classes, grain_size, num_bits, M2D, save_path):
        super(UniteLinear, self).__init__(in_features = infeatures, out_features = classes, bias=True)
        self.grain_size = grain_size
        self.num_bits = num_bits
        self.M2D = M2D
        self.save_path = save_path
        print("FClayer: grain_size: %s, num_bits: %d, M2D ratio: %.4f"% (str(grain_size), num_bits, M2D))
    def forward(self, input):
        weight = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, save_path = self.save_path)(self.weight)
        output = F.linear(input, weight, self.bias)

        return output

class UniteConv2d(nn.Conv2d):

    def __init__(self, inplanes, planes, kernel_size, stride, padding, bias, grain_size, num_bits, M2D, save_path):
        super(UniteConv2d, self).__init__(in_channels = inplanes, out_channels = planes, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)
        self.grain_size = grain_size
        self.num_bits = num_bits
        self.M2D = M2D
        self.save_path = save_path
        print("Convlayer: grain_size: %s, num_bits: %d, M2D ratio: %.4f"% (str(grain_size), num_bits, M2D))

    def forward(self, input):
        weight = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, save_path = self.save_path)(self.weight)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output
