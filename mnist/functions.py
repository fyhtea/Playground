import torch
import torch.nn as nn
from torch.autograd import Function

class BinarizeFunc(Function):
  
  @staticmethod
  def forward(input):
    output = input.new(input.size())
    output[input>=0] = 1
    output[input<0] = -1
    return output

  @staticmethod
  def backward(grad_output):
    grad_input = grad_output.clone()
    return grad_input

# aliases
#binarize = BinarizeFunc.apply
def binarize(input):
	return BinarizeFunc()(input)
