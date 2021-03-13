from __future__ import print_function
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Function

class DtoA_3bit(Function):
    def __init__(self, Vref, sigma):
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
    def convert(self, input):
        output = torch.where(input >= 1.75, self.L7, input)
        output = torch.where((output < 1.75) & (output >= 1.50), self.L6, output)
        output = torch.where((output < 1.50) & (output >= 1.25), self.L5, output)
        output = torch.where((output < 1.25) & (output >= 1.00), self.L4, output)
        output = torch.where((output < 1.00) & (output >= 0.75), self.L3, output)
        output = torch.where((output < 0.75) & (output >= 0.50), self.L2, output)
        output = torch.where((output < 0.50) & (output >= 0.25), self.L1, output)
        output = torch.where(output < 0.25, torch.tensor(0.0), output)
        return output
    def show_DA_tranfunc(self):
        x_in = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype = torch.float32)
        x_out = self.convert(x_in)
        plt.subplot(221)
        plt.plot(x_in.numpy(), x_out.numpy())
        plt.title("Transfer Function of given D/A")
        return None
class AtoD_3bit(Function):
    def __init__(self, Vref, sigma):
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
    def convert(self, input):
        output = torch.where(input >= self.L7, torch.tensor(1.75), input)
        output = torch.where((output < self.L7) & (output >= self.L6), torch.tensor(1.5), output)
        output = torch.where((output < self.L6) & (output >= self.L5), torch.tensor(1.25), output)
        output = torch.where((output < self.L5) & (output >= self.L4), torch.tensor(1.0), output)
        output = torch.where((output < self.L4) & (output >= self.L3), torch.tensor(0.75), output)
        output = torch.where((output < self.L3) & (output >= self.L2), torch.tensor(0.5), output)
        output = torch.where((output < self.L2) & (output >= self.L1), torch.tensor(0.25), output)
        output = torch.where(output < self.L1, torch.tensor(0.0), output)
        return output
    def show_AD_tranfunc(self):
        x_in = torch.from_numpy(np.arange(0, 2, 0.01)).type(torch.float32)
        x_out = self.convert(x_in)
        plt.subplot(222)
        plt.plot(x_in.numpy(), x_out.numpy())
        plt.title("Transfer Function of given A/D")
        return None  

  

  
def AD_MC_test(sigma1, sigma2, shown = False):
    ideal_AtoD =  AtoD_3bit(2.0, 0.0)
    Bob_AtoD = AtoD_3bit(2.0, sigma1)
    if shown == True:
      show_AD_tranfunc(Bob_AtoD)
    Alice_DtoA = DtoA_3bit(2.0, sigma2)
    if shown == True:
      show_DA_tranfunc(Alice_DtoA)
    x = 1 + 0.5* torch.randn(20)
    x_raw = ideal_AtoD.convert(x)
    x_being_affected = Bob_AtoD.convert(Alice_DtoA.convert(x_raw))
    error_rate = torch.nonzero(x_raw - x_being_affected).size()[0]/x.size()[0]
    return error_rate
  
#Test the function
print(AD_MC_test(0.0, 0.0, True))
#Do a sweep
sigma2_list = list(range(8,0, -1))
sigma2_list = [10**(-x) for x in sigma2_list]
sigma1_list = list(range(8,0, -1))
sigma1_list = [10**(-x) for x in sigma1_list]
error_list = []
for t in range(8):
    avg_results = []
    for m in range(1000):
        avg_results.append(100*AD_MC_test(sigma1_list[t], 0.0))
    error_list.append(np.average(avg_results))
plt.subplot(223)
plt.plot(sigma1_list, error_list)
plt.legend("A/D Only")
plt.xscale('log')
error_list = []
for t in range(8):
    avg_results = []
    for m in range(1000):
        avg_results.append(100*AD_MC_test(0.0, sigma2_list[t]))
    error_list.append(np.average(avg_results))
plt.subplot(223)
plt.plot(sigma2_list, error_list)
plt.legend("D/A Only")
plt.xscale('log')
error_list = []
for t in range(8):
    avg_results = []
    for m in range(1000):
        avg_results.append(100*AD_MC_test(sigma1_list[t], sigma2_list[t]))
    error_list.append(np.average(avg_results))
plt.subplot(223)
plt.plot(sigma1_list, error_list)
plt.legend("Both")
plt.xscale('log')
  