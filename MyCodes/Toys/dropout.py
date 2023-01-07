import torch
from torch import nn

m = nn.Dropout(p=0.2)
input = torch.randn(5, 4)
print(input)
output = m(input)
print(output)