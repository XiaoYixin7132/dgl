import torch

x = torch.zeros(1, 4)
y = torch.cat((x, x), dim=0)

print(x)
print(y)