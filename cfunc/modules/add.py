import torch.nn as nn
import functions.add as myadd


class MyAddModule(nn.Module):
  def forward(self, input1, input2):
    return myadd.my_add(input1, input2)
