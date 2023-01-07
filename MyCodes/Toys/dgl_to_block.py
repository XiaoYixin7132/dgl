import dgl
import torch

from dgl import AddSelfLoop
from dgl.data import CoraGraphDataset

transform = (
    AddSelfLoop()
)  # by default, it will first remove self-loops to prevent duplication
data = CoraGraphDataset(transform=transform)
g = data[0]

# g = dgl.graph(([1, 2], [2, 3]))
dst_nodes = torch.tensor([2, 3])
block = dgl.to_block(g, dst_nodes)
print(block)