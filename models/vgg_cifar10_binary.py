import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  get_centroid, get_quantized,DtoA_3bit,AtoD_3bit

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

class VGG_Cifar10_Binary(nn.Module):

    def __init__(self, num_classes=10, input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, 
  res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, 
  output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0,
  save_path = './'):
        super(VGG_Cifar10_Binary, self).__init__()
        self.infl_ratio=3;
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1_3x3 = BinarizeConv2d(3, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1, bias=True, grain_size = input_grain_size, num_bits = input_num_bits, M2D = input_M2D, save_path = save_path)
        self.bn_1 = nn.BatchNorm2d(128*self.infl_ratio)
        self.conv_2 = BinarizeConv2d(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1, bias=True, grain_size = res_grain_size, num_bits = res_num_bits, M2D = res_M2D, save_path = save_path)
        self.bn_2 = nn.BatchNorm2d(128*self.infl_ratio)
        self.conv_3 = BinarizeConv2d(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, stride=1, padding=1, bias=True, grain_size = res_grain_size, num_bits = res_num_bits, M2D = res_M2D, save_path = save_path)
        self.bn_3 = nn.BatchNorm2d(256*self.infl_ratio)
        self.conv_4 = BinarizeConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, stride=1, padding=1, bias=True, grain_size = res_grain_size, num_bits = res_num_bits, M2D = res_M2D, save_path = save_path)
        self.bn_4 = nn.BatchNorm2d(256*self.infl_ratio)
        self.conv_5 = BinarizeConv2d(256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, stride=1, padding=1, bias=True, grain_size = res_grain_size, num_bits = res_num_bits, M2D = res_M2D, save_path = save_path)
        self.bn_5 = nn.BatchNorm2d(512*self.infl_ratio)
        self.conv_6 = BinarizeConv2d(512*self.infl_ratio, 512, kernel_size=3, stride=1, padding=1, bias=True, grain_size = res_grain_size, num_bits = res_num_bits, M2D = res_M2D, save_path = save_path)
        self.bn_6 = nn.BatchNorm2d(512)
        
        self.linear_7 = BinarizeLinear(512 * 4 * 4, 1024, grain_size = output_grain_size, num_bits = output_num_bits, M2D = output_M2D, save_path = save_path)
        self.bn_7 = nn.BatchNorm1d(1024)
        
        self.linear_8 = BinarizeLinear(1024, 1024, grain_size = output_grain_size, num_bits = output_num_bits, M2D = output_M2D, save_path = save_path)
        self.bn_8 = nn.BatchNorm1d(1024)
        
        self.linear_9 = BinarizeLinear(1024, num_classes, grain_size = output_grain_size, num_bits = output_num_bits, M2D = output_M2D, save_path = save_path)
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
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)
        x = self.conv_1_3x3(x)
        x = self.bn_1(x)
        x = AtoD_3bit(2.0, 0)(x)
        x = self.conv_2(x)
        x = self.maxpool(x)
        x = self.bn_2(x)
        x = AtoD_3bit(2.0, 0)(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = AtoD_3bit(2.0, 0)(x)
        x = self.conv_4(x)
        x = self.maxpool(x)
        x = self.bn_4(x)
        x = AtoD_3bit(2.0, 0)(x)
        x = self.conv_5(x)
        x = self.bn_5(x)
        x = AtoD_3bit(2.0, 0)(x)
        x = self.conv_6(x)
        x = self.maxpool(x)
        x = self.bn_6(x)
        x = AtoD_3bit(2.0, 0)(x)
        x = x.view(-1, 512 * 4 * 4)
        
        x = self.linear_7(x)
        x = self.bn_7(x)
        x = AtoD_3bit(2.0, 0)(x)
        x = self.linear_8(x)
        x = self.bn_8(x)
        x = AtoD_3bit(2.0, 0)(x)
        x = self.linear_9(x)
        return x


def vgg_cifar10_binary(num_classes, input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0, save_path = './', **kwargs):
    global ti
    ti = 0
    global num_res
    num_res = 9
    return VGG_Cifar10_Binary(num_classes, input_grain_size, input_num_bits, input_M2D, res_grain_size, res_num_bits, res_M2D, output_grain_size, output_num_bits, output_M2D, save_path)
