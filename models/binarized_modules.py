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

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

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
        
class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

class Binarize(torch.autograd.Function):
    def __init__(self, grain_size, num_bits, M2D, save_path):
        super(Binarize,self).__init__()
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
        output = self.W * input_d.sign()
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

class BinarizeLinear(nn.Linear):

    def __init__(self, infeatures, classes, grain_size, num_bits, M2D, save_path):
        super(BinarizeLinear, self).__init__(in_features = infeatures, out_features = classes, bias=True)
        self.grain_size = grain_size
        self.num_bits = num_bits
        self.M2D = M2D
        self.save_path = save_path
        print("FClayer: grain_size: %s, num_bits: %d, M2D ratio: %.4f"% (str(grain_size), num_bits, M2D))
    def forward(self, input):
        weight = Binarize(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, save_path = self.save_path)(self.weight)
        output = F.linear(input, weight, self.bias)

        return output

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, inplanes, planes, kernel_size, stride, padding, bias, grain_size, num_bits, M2D, save_path):
        super(BinarizeConv2d, self).__init__(in_channels = inplanes, out_channels = planes, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)
        self.grain_size = grain_size
        self.num_bits = num_bits
        self.M2D = M2D
        self.save_path = save_path
        print("Convlayer: grain_size: %s, num_bits: %d, M2D ratio: %.4f"% (str(grain_size), num_bits, M2D))

    def forward(self, input):
        weight = Binarize(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, save_path = self.save_path)(self.weight)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output
