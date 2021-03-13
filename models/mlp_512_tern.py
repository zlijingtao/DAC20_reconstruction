from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .binarized_modules import  get_centroid, get_quantized, DtoA_3bit, AtoD_3bit

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

class MLP_4(nn.Module):
    def __init__(self, num_classes, 
          input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, 
          res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, 
          output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0,
          save_path = './'):
        super(MLP_4, self).__init__()
        
        self.fc1 = quanLinear(784, 512, grain_size = input_grain_size, num_bits = input_num_bits, M2D = input_M2D, save_path = save_path)
        self.bn_1 = nn.BatchNorm1d(512, eps=1e-4, momentum=0.15)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = quanLinear(512, 512, grain_size = res_grain_size, num_bits = res_num_bits, M2D = res_M2D, save_path = save_path)
        self.bn_2 = nn.BatchNorm1d(512, eps=1e-4, momentum=0.15)
        self.fc3 = quanLinear(512, 512, grain_size = res_grain_size, num_bits = res_num_bits, M2D = res_M2D, save_path = save_path)
        self.bn_3 = nn.BatchNorm1d(512, eps=1e-4, momentum=0.15)
        self.classifier = quanLinear(512, num_classes, output_grain_size, output_num_bits, output_M2D, save_path = save_path)

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

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        return self.classifier(x)
        
def mlp_512_tern(num_classes=10, input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0, save_path = './'):
  """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  global ti
  ti = 0
  global num_res
  num_res = 4
  model = MLP_4(num_classes, input_grain_size, input_num_bits, input_M2D, res_grain_size, res_num_bits, res_M2D, output_grain_size, output_num_bits, output_M2D, save_path)
  return model