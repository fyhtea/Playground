import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from functions import *

class BinaryTanh(nn.Module):

  def __init__(self):
    super(BinaryTanh, self).__init__()
    self.hardtanh = nn.Hardtanh()

  def forward(self, input):
    output = self.hardtanh(input)
    output = binarize(output)
    return output


class BinaryLinear(nn.Linear):

  def forward(self, input):
    binary_weight =  binarize(self.weight)
    if self.bias is None:
      return F.linear(input, binary_weight)
    else:
      return F.linear(input, binary_weight, self.bias)

  def reset_parameters(self):
    in_features, out_features = self.weight.size()
    stdv = math.sqrt(1.5/(in_features+out_features))
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
      self.bias.data.zero_()
    
    self.weight.lr_scale = 1./stdv

class BinaryConv2d(nn.Conv2d):

  def forward(self, input):
    bw = binarize(self.weight)
    return F.conv2d(input, bw, self.bias, self.stride,
                     self.padding, self.dilation, self.groups)

  def reset_parameters(self):
    in_features = self.in_channels
    out_features = self.out_channels
    for k in self.kernel_size:
      in_features *=k
      out_features *=k
    stdv = math.sqrt(1.5 / (in_features + out_features))
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
      self.bias.data.zero_()
    
    self.weight.lr_scale = 1. / stdv


class QuantizeLinear(nn.Linear):

  def forward(self, input):

    quantized_weight = torch.tanh(self.weight)
    quantized_weight = quantized_weight/torch.max(torch.abs(quantized_weight)).data[0]*0.5+0.5

    quantized_weight = 2*quantize(quantized_weight)-1

    if self.bias is not None:
      quantized_bias = torch.tanh(self.bias)
      quantized_bias = quantized_bias/torch.max(torch.abs(quantized_bias)).data[0]*0.5 + 0.5
      quantized_bias = 2*quantize(quantized_bias)-1

      return F.linear(input, quantized_weight,quantized_bias)
    else:
      return F.linear(input, quantized_weight)


class QuantizeConv2d(nn.Conv2d):

  def forward(self, input):
    qw = quantize(self.weight)
    qb = quantize(self.bias)
    return F.conv2d(input, qw, qb, self.stride,
                    self.padding, self.dilation, self.groups)


class QuantizeActivation(nn.Module):

  def __init__(self):
    super(QuantizeActivation, self).__init__()
    self.hardtanh = nn.Hardtanh()

  def forward(self, input):
    output = (self.hardtanh(input)+1.0)/2
    output = quantize(output)
    return output