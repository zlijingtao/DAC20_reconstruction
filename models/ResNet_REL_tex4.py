import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from torch.nn import init

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

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
    qmin = -(2.**(num_bits - 1) - 1)
    qmax = qmin + 2.**num_bits - 2.
    scale = 2 * M2D / (qmax - qmin)
    output.div_(scale)
    output.clamp_(qmin, qmax).round_()
    output.mul_(scale)
    return output

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
        self.th = self.tFactor*max_w #threshold
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
        tfactor_list = [0.05, 0.1, 0.15, 0.2]
        weight = _quanFunc(tfactor=tfactor_list[0], grain_size = self.grain_size , num_bits = self.num_bits, M2D = self.M2D, save_path = self.save_path)(self.weight)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        for tfactor in tfactor_list[1:]:
            weight = _quanFunc(tfactor=tfactor, grain_size = self.grain_size, num_bits = self.num_bits, M2D = self.M2D, save_path = self.save_path)(self.weight)
            output += F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
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
        tfactor_list = [0.05, 0.1, 0.15, 0.2]
        weight = _quanFunc(tfactor=tfactor_list[0], grain_size = self.grain_size, num_bits = self.num_bits, M2D = self.M2D, save_path = self.save_path)(self.weight)
        output = F.linear(input, weight, self.bias)
        
        for tfactor in tfactor_list[1:]:
            weight = _quanFunc(tfactor=tfactor, grain_size = self.grain_size, num_bits = self.num_bits, M2D = self.M2D, save_path = self.save_path)(self.weight)
            output += F.linear(input, weight, self.bias)
        return output

        return output

def conv3x3(in_planes, out_planes, stride=1, grain_size = (1,1), num_bits = 4, M2D = 0, save_path = './'):
    """3x3 convolution with padding"""
    return quanConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, grain_size = grain_size, num_bits = num_bits, M2D = M2D, save_path = save_path)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, grain_size = (1,1), num_bits = 4, M2D = 0, save_path = './'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, grain_size, num_bits, M2D, save_path)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1, grain_size, num_bits, M2D, save_path)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, grain_size = (1,1), num_bits = 4, M2D = 0, save_path = './'):
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1 = quanConv2d(inplanes, planes, kernel_size=1, stride=1, 
                                padding=0, bias=False, grain_size = grain_size, num_bits = num_bits, M2D = M2D, save_path = save_path)

        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        self.conv2 = quanConv2d(planes, planes, kernel_size=3, stride=stride, 
                                padding=1, bias=False, grain_size = grain_size, num_bits = num_bits, M2D = M2D, save_path = save_path)

        self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.conv3 = quanConv2d(planes, planes * self.expansion, kernel_size=1, stride=1, 
                                padding=0, bias=False, grain_size = grain_size, num_bits = num_bits, M2D = M2D, save_path = save_path)
        
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, fp_fl=True, fp_ll=True, 
  input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, 
  res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, 
  output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0,
  save_path = './'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if fp_fl:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = quanConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, grain_size = input_grain_size, num_bits = input_num_bits, M2D = input_M2D, save_path = save_path)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, res_grain_size, res_num_bits ,res_M2D, save_path = save_path)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, res_grain_size, res_num_bits ,res_M2D, save_path = save_path)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, res_grain_size, res_num_bits ,res_M2D, save_path = save_path)
        self.layer4 = self._make_layer(block, 512, layers[3], 2, res_grain_size, res_num_bits ,res_M2D, save_path = save_path)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if fp_ll:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = quanLinear(512 * block.expansion, num_classes, output_grain_size, output_num_bits, output_M2D, save_path = save_path)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, grain_size = (1,1), num_bits = 4, M2D = 0, save_path = './'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                quanConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,
                          grain_size = grain_size, num_bits = num_bits, M2D = M2D, save_path = save_path),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, grain_size, num_bits, M2D, save_path))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, grain_size, num_bits, M2D, save_path))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# def quan_resnet18b_ff_lf(num_classes=1000):
#     model = ResNet(BasicBlock, [2, 2, 2, 2], fp_fl=True, fp_ll=True)
#     return model

def resnet18b_fq_lq_tern_tex_4(num_classes=1000, input_grain_size = (1,1), input_num_bits = 4, input_M2D = 0.0, res_grain_size = (1,1), res_num_bits = 4, res_M2D = 0.0, output_grain_size = (1,1), output_num_bits = 4, output_M2D = 0.0, save_path = './'):
    global ti
    ti = 0
    global num_res
    num_res = 18
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, False, False, input_grain_size, input_num_bits, input_M2D, res_grain_size, res_num_bits, res_M2D, output_grain_size, output_num_bits, output_M2D, save_path)
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

