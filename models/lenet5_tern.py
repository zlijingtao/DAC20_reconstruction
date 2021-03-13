from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .ternarized_modules import  get_centroid, get_quantized, DtoA_3bit, AtoD_3bit

class _quanFunc(torch.autograd.Function):

    def __init__(self, tfactor, grain_size, num_bits, M2D, save_path):
        super(_quanFunc,self).__init__()
        self.tFactor = tfactor
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
        max_w = input_d.abs().max()
        self.th = self.tFactor * max_w #threshold
        output = input.clone().zero_()
        self.W = 1-self.M2D
        output[input_d.ge(self.th)] = self.W
        output[input_d.lt(-self.th)] = -self.W
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


class quanConv2d(nn.Conv2d):
    def __init__(self, inplanes, planes, kernel_size, stride, padding, bias, grain_size, num_bits, M2D, save_path):
        super(quanConv2d, self).__init__(in_channels = inplanes, out_channels = planes, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)
        self.grain_size = grain_size
        self.num_bits = num_bits
        self.M2D = M2D
        self.save_path = save_path
        print("Convlayer: grain_size: %s, num_bits: %d, M2D ratio: %.4f"% (str(grain_size), num_bits, M2D))
    def forward(self, input):
        tfactor_list = [0.05]
        weight = _quanFunc(tfactor=tfactor_list[0], grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, save_path = self.save_path)(self.weight)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output 

class quanLinear(nn.Linear):
    def __init__(self, infeatures, classes, grain_size, num_bits, M2D, save_path):
        super(quanLinear, self).__init__(in_features = infeatures, out_features = classes, bias=True)
        self.grain_size = grain_size
        self.num_bits = num_bits
        self.M2D = M2D
        self.save_path = save_path
        print("FClayer: grain_size: %s, num_bits: %d, M2D ratio: %.4f"% (str(grain_size), num_bits, M2D))
    def forward(self, input):
        tfactor_list = [0.05]
        weight = _quanFunc(tfactor=tfactor_list[0], grain_size = self.grain_size, num_bits = self.num_bits, M2D = self.M2D, save_path = self.save_path)(self.weight)
        output = F.linear(input, weight, self.bias)

        return output

class LeNet_5(nn.Module):
    def __init__(self, num_classes, 
          input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, 
          res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, 
          output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0,
          save_path = './'):
        super(LeNet_5, self).__init__()
        
        self.conv1 = quanConv2d(1, 20, kernel_size=5, stride=1, padding=1, bias=False, grain_size = input_grain_size, num_bits = input_num_bits, M2D = input_M2D, save_path = save_path)
        self.bn_conv1 = nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bin_conv2 = quanConv2d(20, 50, kernel_size=5, stride=1, padding=0, bias=False, grain_size = res_grain_size, num_bits = res_num_bits, M2D = res_M2D, save_path = save_path)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bin_ip1 = quanConv2d(50*4*4, 500, kernel_size=3, stride=1, padding=1, bias=False, grain_size = res_grain_size, num_bits = res_num_bits, M2D = res_M2D, save_path = save_path)
        
        self.ip2 = quanLinear(500, num_classes, output_grain_size, output_num_bits, output_M2D, save_path = save_path)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)
        return

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)
        x = self.bin_conv2(x)
        x = self.pool2(x)

        # x = x.view(x.size(0), 50*4*4)

        x = self.bin_ip1(x)
        x = self.ip2(x)
        return x
        
def lenet5_tern(num_classes=10, input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0, save_path = './'):
  """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  global ti
  ti = 0
  global num_res
  num_res = 4
  model = LeNet_5(num_classes, input_grain_size, input_num_bits, input_M2D, res_grain_size, res_num_bits, res_M2D, output_grain_size, output_num_bits, output_M2D, save_path)
  return model