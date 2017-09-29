import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.add import MyAddModule



class MyNetwork(nn.Module):
  def __init__(self):
    super(MyNetwork, self).__init__()
    self.add = MyAddModule()

  def forward(self, input1, input2):
    return self.add(input1, input2)



model = MyNetwork()

x = torch.arange(1,25).view(1,24)
input1, input2 = Variable(x), Variable(x*2)

print ('test for c_func_forward:')
print (model(input1, input2))
print (input1+input2)

if torch.cuda.is_available():
  print ('test for cuda_func_forward')
  input1, input2 = input1.cuda(), input2.cuda()
  print(model(input1,input2))
  print(input1+input2)


