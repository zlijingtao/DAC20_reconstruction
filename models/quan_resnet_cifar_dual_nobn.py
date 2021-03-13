import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np
from .fixpoint_modules import  get_centroid, get_quantized, quantize, get_aggregate
from .check_modules import get_mean_bits
from os import path


class Unite(torch.autograd.Function):
    def __init__(self, grain_size, num_bits, M2D, step_size, half_lvls, only_cent, only_dev, save_path):
        super(Unite,self).__init__()
        self.grain_size = grain_size #grain size in tuple
        self.M2D = M2D
        self.num_bits = num_bits
        self.step_size = step_size
        self.half_lvls = half_lvls
        self.save_path = save_path
        self.only_cent = only_cent
        self.only_dev = only_dev
    def forward(self, input):
        self.save_for_backward(input)
        input = input/self.step_size

        centroid = get_centroid(input, self.grain_size, self.num_bits, self.M2D, self.half_lvls)
        global ti
        global num_res
        ti += 1

        input_d = (input - centroid)
        output = input.clone().zero_()
        self.W = (1-self.M2D) * self.half_lvls

        output = input_d.clamp_(-self.W, self.W).round_()
        if ti <=num_res:
            torch.save(centroid* self.step_size, self.save_path + '/saved_tensors/centroid{}.pt'.format(ti))
            torch.save(output* self.step_size, self.save_path + '/saved_tensors/deviation{}.pt'.format(ti))
        if self.only_cent:
            output = centroid
        elif self.only_dev == False:
            output = output + centroid

        return output

    def backward(self, grad_output):
        grad_input = grad_output.clone() / self.step_size
        input, = self.saved_tensors
        grad_input[input.ge(self.half_lvls * self.step_size.item())] = 0
        grad_input[input.le(-self.half_lvls * self.step_size.item())] = 0
        return grad_input

class quan_Conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 grain_size = (1,1), num_bits = 3, M2D = 0.0, save_path = './'):
        super(quan_Conv2d, self).__init__(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)
        self.N_bits = 8
        self.grain_size = grain_size
        self.num_bits = num_bits
        self.M2D = M2D
        self.save_path = save_path
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1.0 / self.half_lvls]), requires_grad=False)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  #in-place change MSB to negative

    def forward(self, input, dual_module = False, running_mean = 0, running_var = 0, beta = 0, gamma = 0):

        if self.inf_with_weight:
            #Option 1: Robust case
            weight_centroid = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, only_cent = True, only_dev = False, save_path = self.save_path)(self.weight * self.step_size)
            weight_dev = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, only_cent = False, only_dev = True, save_path = self.save_path)(self.weight * self.step_size)
            #Option 2: Vulnerable case
            
            centroid_activation = F.conv2d(input, weight_centroid * self.step_size, self.bias,
                            self.stride, self.padding, self.dilation,
                            self.groups)
            dev_activation = F.conv2d(input, weight_dev * self.step_size, self.bias,
                            self.stride, self.padding, self.dilation,
                            self.groups)

            if dual_module == True:
                full_activation = get_aggregate(centroid_activation, dev_activation, running_mean, running_var, gamma, beta)
            else:
                full_activation = centroid_activation + dev_activation
            # del dev_activation, centroid_activation, weight_centroid, weight_dev
            return full_activation
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size,
                                   self.half_lvls) * self.step_size
            weight_centroid = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, only_cent = True, only_dev = False, save_path = self.save_path)(weight_quan)* self.step_size
            weight_dev = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, only_cent = False, only_dev = True, save_path = self.save_path)(weight_quan)* self.step_size

            centroid_activation = F.conv2d(input, weight_centroid, self.bias,
                            self.stride, self.padding, self.dilation,
                            self.groups)
            dev_activation = F.conv2d(input, weight_dev, self.bias,
                            self.stride, self.padding, self.dilation,
                            self.groups)
            if dual_module == True:
                full_activation = get_aggregate(centroid_activation, dev_activation, running_mean, running_var, gamma, beta)
            else:
                full_activation = centroid_activation + dev_activation
            # del dev_activation, centroid_activation, weight_centroid, weight_dev
            return full_activation

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = torch.tensor(1.0) / self.half_lvls
            # self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            #Option 1: Robust case
            self.weight.data = quantize(self.weight, self.step_size, self.half_lvls)
            #Option 2: Vulnerable case
            # weight_quan = quantize(self.weight, self.step_size, self.half_lvls)* self.step_size
            # self.weight.data = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, save_path = self.save_path)(weight_quan)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True


class quan_Linear(nn.Linear):
    def __init__(self, in_features, out_features, grain_size = (1,1), num_bits = 3, M2D = 0.0, save_path = './'):
        super(quan_Linear, self).__init__(in_features, out_features, bias=True)
        self.N_bits = 8
        self.grain_size = grain_size
        self.num_bits = num_bits
        self.M2D = M2D
        self.save_path = save_path
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1.0 / self.half_lvls]), requires_grad=False)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  #in-place reverse

    def forward(self, input, dual_module = False):
        if self.inf_with_weight:
            #Option 1: Robust case
            weight_centroid = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, only_cent = True, only_dev = False, save_path = self.save_path)(self.weight * self.step_size)
            weight_dev = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, only_cent = False, only_dev = True, save_path = self.save_path)(self.weight * self.step_size)
            #Option 2: Vulnerable case
            centroid_activation = F.linear(input, weight_centroid * self.step_size, self.bias)
            dev_activation = F.linear(input, weight_dev * self.step_size, self.bias)
            if dual_module == True:
                full_activation = get_aggregate(centroid_activation, dev_activation, 0, 0, 0, 0)
            else:
                full_activation = centroid_activation + dev_activation
            # del dev_activation, centroid_activation, weight_centroid, weight_dev
            # full_activation = get_aggregate(centroid_activation, dev_activation, 0, 0, 0, 0)
            return full_activation
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size,
                                   self.half_lvls) * self.step_size
            weight_centroid = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, only_cent = True, only_dev = False, save_path = self.save_path)(weight_quan)* self.step_size
            weight_dev = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, only_cent = False, only_dev = True, save_path = self.save_path)(weight_quan)* self.step_size
            centroid_activation = F.linear(input, weight_centroid, self.bias)
            dev_activation = F.linear(input, weight_dev, self.bias)
            if dual_module == True:
                full_activation = get_aggregate(centroid_activation, dev_activation, 0, 0, 0, 0)
            else:
                full_activation = centroid_activation + dev_activation
            # del dev_activation, centroid_activation, weight_centroid, weight_dev
            return full_activation

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = torch.tensor(1.0) / self.half_lvls
            # self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            #Option 1: Robust case
            self.weight.data = quantize(self.weight, self.step_size, self.half_lvls)
            #Option 2: Vulnerable case
            # weight_quan = quantize(self.weight, self.step_size, self.half_lvls)* self.step_size
            # self.weight.data = Unite(grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, step_size = self.step_size, half_lvls = self.half_lvls, save_path = self.save_path)(weight_quan)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True

class DownsampleA(nn.Module):

  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__()
    assert stride == 2
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

  def forward(self, x):
    x = self.avg(x)
    return torch.cat((x, x.mul(0)), 1)

class ResNetBasicblock(nn.Module):
  expansion = 1
  """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
  def __init__(self, inplanes, planes, stride=1, downsample=None, grain_size = (1,1), num_bits = 4, M2D = 0, save_path = './'):
    super(ResNetBasicblock, self).__init__()

    self.conv_a = quan_Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, grain_size = grain_size, num_bits = num_bits, M2D = M2D, save_path = save_path)

    self.conv_b = quan_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, grain_size = grain_size, num_bits = num_bits, M2D = M2D, save_path = save_path)

    self.downsample = downsample

  def forward(self, x):
    residual = x

    basicblock = self.conv_a(x, dual_module = True)
    basicblock = F.relu(basicblock, inplace=True)

    basicblock = self.conv_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(x)

    return F.relu(residual + basicblock, inplace=True)

class CifarResNet(nn.Module):
  """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
  def __init__(self, block, depth, num_classes,
  input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0,
  res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0,
  output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0,
  save_path = './'):
    """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      grain: grain size as tuple
      M2D: Mean to Deviation ratio
      base_width: base width
    """
    super(CifarResNet, self).__init__()

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
    layer_blocks = (depth - 2) // 6
    print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

    self.num_classes = num_classes

    self.conv_1_3x3 = quan_Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, grain_size = input_grain_size, num_bits = input_num_bits, M2D = input_M2D, save_path = save_path)

    self.inplanes = 16
    self.stage_1 = self._make_layer(block, 16, layer_blocks, 1, res_grain_size, res_num_bits ,res_M2D, save_path = save_path)
    self.stage_2 = self._make_layer(block, 32, layer_blocks, 2, res_grain_size, res_num_bits ,res_M2D, save_path = save_path)
    self.stage_3 = self._make_layer(block, 64, layer_blocks, 2, res_grain_size, res_num_bits ,res_M2D, save_path = save_path)
    self.avgpool = nn.AvgPool2d(8)
    self.classifier = quan_Linear(64*block.expansion, num_classes, output_grain_size, output_num_bits, output_M2D, save_path = save_path)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, grain_size = (1,1), num_bits = 4, M2D = 0, save_path = './'):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, grain_size, num_bits, M2D, save_path = save_path))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, 1, None, grain_size, num_bits, M2D, save_path = save_path))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv_1_3x3(x, dual_module = False)
    x = F.relu(x, inplace=True)
    x = self.stage_1(x)
    x = self.stage_2(x)
    x = self.stage_3(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x, dual_module = False)
    return x

def dual_quan_resnet20_nobn(num_classes=10, input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0, save_path = './'):
  """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  global ti
  ti = 0
  global num_res
  num_res = 20
  model = CifarResNet(ResNetBasicblock, 20, num_classes, input_grain_size, input_num_bits, input_M2D, res_grain_size, res_num_bits, res_M2D, output_grain_size, output_num_bits, output_M2D, save_path)
  return model

def dual_quan_resnet32_nobn(num_classes=10, input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0, save_path = './'):
  """Constructs a ResNet-32 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  global ti
  ti = 0
  global num_res
  num_res = 32
  model = CifarResNet(ResNetBasicblock, 32, num_classes, input_grain_size, input_num_bits, input_M2D, res_grain_size, res_num_bits, res_M2D, output_grain_size, output_num_bits, output_M2D, save_path)
  return model

def dual_quan_resnet44_nobn(num_classes=10, input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0, save_path = './'):
  """Constructs a ResNet-44 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  global ti
  ti = 0
  global num_res
  num_res = 44
  model = CifarResNet(ResNetBasicblock, 44, num_classes, input_grain_size, input_num_bits, input_M2D, res_grain_size, res_num_bits, res_M2D, output_grain_size, output_num_bits, output_M2D, save_path)
  return model

def dual_quan_resnet56_nobn(num_classes=10, input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0, save_path = './'):
  """Constructs a ResNet-56 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  global ti
  ti = 0
  global num_res
  num_res = 56
  model = CifarResNet(ResNetBasicblock, 56, num_classes, input_grain_size, input_num_bits, input_M2D, res_grain_size, res_num_bits, res_M2D, output_grain_size, output_num_bits, output_M2D, save_path)
  return model

def dual_quan_resnet110_nobn(num_classes=10, input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0, save_path = './'):
  """Constructs a ResNet-110 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  global ti
  ti = 0
  global num_res
  num_res = 110
  model = CifarResNet(ResNetBasicblock, 110, num_classes, input_grain_size, input_num_bits, input_M2D, res_grain_size, res_num_bits, res_M2D, output_grain_size, output_num_bits, output_M2D, save_path)
  return model
