import torch
import torch.nn as nn
import torch.nn.functional as F


import dgl.nn as dglnn

from dgl import add_reverse_edges, add_edges
from dgl import graph, NID, EID



torch.manual_seed(12345)
class GNNModel(nn.Module):
    def __init__(self,  gnn_layer: str, n_layers: int, layer_dim: int,
                 input_feature_dim: int, n_classes: int):
        super().__init__()

        assert n_layers >= 1, 'GNN must have at least one layer'
        dims = [input_feature_dim] + [layer_dim] * (n_layers-1) + [n_classes]

        self.convs = nn.ModuleList()
        for idx in range(len(dims) - 1):
            if gnn_layer == 'gat':
                # use 2 aattention heads
                layer = dglnn.GATConv(dims[idx], dims[idx+1], 1)  # pylint: disable=no-member
            elif gnn_layer == 'gcn':
                layer = dglnn.GraphConv(dims[idx], dims[idx+1], allow_zero_in_degree=True)  # pylint: disable=no-member
            elif gnn_layer == 'sage':
                # Use mean aggregtion
                # pylint: disable=no-member
                layer = dglnn.SAGEConv(dims[idx], dims[idx+1],
                                        aggregator_type='mean')
            else:
                raise ValueError(f'unknown gnn layer type {gnn_layer}')
            self.convs.append(layer)
     
    def forward(self, graph, features, edge_weight=None):
        for idx, conv in enumerate(self.convs):
            features = conv(graph, features, edge_weight=edge_weight)
            if features.ndim == 3:  # GAT produces an extra n_heads dimension
                # collapse the n_heads dimension
                features = features.mean(1)

            if idx < len(self.convs) - 1:
                features = F.relu(features, inplace=True)

        return features       
    

if __name__ == "__main__":
    
    g = graph((torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]), torch.tensor([1, 2, 0, 3, 0, 3, 1, 2])))

    features = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    model = GNNModel('gcn', 1, 16, features.size(1), 4).to(device)
    
    # logits = model(g.to(device), features.to(device))

    for i in range(2):
        if i == 1:
            g = add_edges(g, torch.tensor([0]), torch.tensor([0]))
        print(g.edges())
        logits = model(g.to(device), features.to(device))
        print(logits)