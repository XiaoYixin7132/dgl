import torch

x = torch.randn(3, 4)
print(x)
indices = torch.tensor([0, 2])
print(x.index_select(0, indices))
reverse_indices = torch.tensor([2, 0])
print(x.index_select(0, reverse_indices))