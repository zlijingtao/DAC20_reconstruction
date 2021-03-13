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
        original_size = input.size()
        reshaped_input = input.view(1, 1, original_size[0], original_size[1])
        pooling_result = F.avg_pool2d(reshaped_input, grain_size, grain_size)
        pooling_result = get_quantized(pooling_result, num_bits, M2D)
        pooling_result = pooling_result.view(pooling_result.size()[2:])
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
        qmin = -(2.**(num_bits - 1) - 1)
        qmax = qmin + 2.**num_bits - 2.
        scale = 2 * M2D / (qmax - qmin)
        output.div_(scale)
        output.clamp_(qmin, qmax).round_()
        output.mul_(scale)
    else:
        output = input.clone().zero_()
    return output

class DtoA_3bit(torch.autograd.Function):
    def __init__(self, Vref, sigma):
        super(DtoA_3bit,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Vref = Vref
        self.sigma = sigma
        self.x = 1 + sigma* torch.randn(8)
        self.x_total = torch.sum(self.x)
        self.x4 = self.x[0] + self.x[1] + self.x[2] + self.x[3]
        self.x2 = self.x[4] + self.x[5]
        self.x1 = self.x[6]
        self.L7 = (1/self.x_total)* self.Vref * self.x4 + (1/self.x_total)* self.Vref * self.x2 + (1/self.x_total)* self.Vref * self.x1
        self.L6 = (1/self.x_total)* self.Vref * self.x4 + (1/self.x_total)* self.Vref * self.x2
        self.L5 = (1/self.x_total)* self.Vref * self.x4 + (1/self.x_total)* self.Vref * self.x1
        self.L4 = (1/self.x_total)* self.Vref * self.x4
        self.L3 = (1/self.x_total)* self.Vref * self.x2 + (1/self.x_total)* self.Vref * self.x1
        self.L2 = (1/self.x_total)* self.Vref * self.x2
        self.L1 = (1/self.x_total)* self.Vref * self.x1
        self.L1 = self.L1.to(self.device)
        self.L2 = self.L2.to(self.device)
        self.L3 = self.L3.to(self.device)
        self.L4 = self.L4.to(self.device)
        self.L5 = self.L5.to(self.device)
        self.L6 = self.L6.to(self.device)
        self.L7 = self.L7.to(self.device)
    def forward(self, input):
        self.save_for_backward(input)
        output = torch.where(input >= 1.75, self.L7, input)
        output = torch.where((output < 1.75) & (output >= 1.50), self.L6, output)
        output = torch.where((output < 1.50) & (output >= 1.25), self.L5, output)
        output = torch.where((output < 1.25) & (output >= 1.00), self.L4, output)
        output = torch.where((output < 1.00) & (output >= 0.75), self.L3, output)
        output = torch.where((output < 0.75) & (output >= 0.50), self.L2, output)
        output = torch.where((output < 0.50) & (output >= 0.25), self.L1, output)
        output = torch.where(output < 0.25, torch.tensor(0.0), output)
        return output
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(2.0)] = 0
        grad_input[input.le(0)] = 0
        return grad_input
        
class AtoD_3bit(torch.autograd.Function):
    def __init__(self, Vref, sigma):
        super(AtoD_3bit,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Vref = Vref
        self.sigma = sigma
        self.x = 1 + sigma* torch.randn(8)
        self.x_total = torch.sum(self.x)
        self.x4 = self.x[0] + self.x[1] + self.x[2] + self.x[3]
        self.x2 = self.x[4] + self.x[5]
        self.x1 = self.x[6]
        self.L7 = (1/self.x_total)* self.Vref * self.x4 + (1/self.x_total)* self.Vref * self.x2 + (1/self.x_total)* self.Vref * self.x1
        self.L6 = (1/self.x_total)* self.Vref * self.x4 + (1/self.x_total)* self.Vref * self.x2
        self.L5 = (1/self.x_total)* self.Vref * self.x4 + (1/self.x_total)* self.Vref * self.x1
        self.L4 = (1/self.x_total)* self.Vref * self.x4
        self.L3 = (1/self.x_total)* self.Vref * self.x2 + (1/self.x_total)* self.Vref * self.x1
        self.L2 = (1/self.x_total)* self.Vref * self.x2
        self.L1 = (1/self.x_total)* self.Vref * self.x1
        self.L1 = self.L1.to(self.device)
        self.L2 = self.L2.to(self.device)
        self.L3 = self.L3.to(self.device)
        self.L4 = self.L4.to(self.device)
        self.L5 = self.L5.to(self.device)
        self.L6 = self.L6.to(self.device)
        self.L7 = self.L7.to(self.device)
    def forward(self, input):
        self.save_for_backward(input)
        output = torch.where(input >= self.L7, torch.tensor(1.75).cuda(), input)
        output = torch.where((output < self.L7) & (output >= self.L6), torch.tensor(1.5).cuda(), output)
        output = torch.where((output < self.L6) & (output >= self.L5), torch.tensor(1.25).cuda(), output)
        output = torch.where((output < self.L5) & (output >= self.L4), torch.tensor(1.0).cuda(), output)
        output = torch.where((output < self.L4) & (output >= self.L3), torch.tensor(0.75).cuda(), output)
        output = torch.where((output < self.L3) & (output >= self.L2), torch.tensor(0.5).cuda(), output)
        output = torch.where((output < self.L2) & (output >= self.L1), torch.tensor(0.25).cuda(), output)
        output = torch.where(output < self.L1, torch.tensor(0.0).cuda(), output)
        return output
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(2.0)] = 0
        grad_input[input.le(0)] = 0
        return grad_input
        