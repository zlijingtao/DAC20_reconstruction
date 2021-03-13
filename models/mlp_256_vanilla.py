from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class MLP_4(nn.Module):
    def __init__(self, num_classes):
        super(MLP_4, self).__init__()
        
        self.fc1 = nn.Linear(784, 256)
        self.bn_1 = nn.BatchNorm1d(256, eps=1e-4, momentum=0.15)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 256)
        self.bn_2 = nn.BatchNorm1d(256, eps=1e-4, momentum=0.15)
        self.fc3 = nn.Linear(256, 256)
        self.bn_3 = nn.BatchNorm1d(256, eps=1e-4, momentum=0.15)
        self.classifier = nn.Linear(256, num_classes)

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
        
def mlp_256(num_classes=10):
  """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = MLP_4(num_classes)
  return model