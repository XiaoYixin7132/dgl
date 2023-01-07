import dgl
import torch
import sys
g = dgl.graph((torch.tensor([0, 1, 2, 3, 4]), torch.tensor([2, 2, 3, 2, 3])))
g.ndata['h'] = torch.arange(g.num_nodes() * 2).view(g.num_nodes(), 2)

rg = dgl.reorder_graph(g, node_permute_algo='metis', permute_config={'k':2})

sys.exit()