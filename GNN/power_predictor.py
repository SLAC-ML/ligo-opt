import torch_geometric as pyg
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.conv import TransformerConv as GATv2Conv

# from fastkan import FastKAN as KAN

class LinGNN(torch.nn.Module):
    def __init__(self, num_features=3, hidden_size=700, target_size=1, num_edge_features=2, num_layers=20, lin_layers = 6, lin_size=800):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.num_layers = num_layers
        self.lin_size = lin_size
        
        self.bnn = BatchNorm(in_channels=num_features)
        self.bne = BatchNorm(in_channels=num_edge_features)
        self.batch_norms = nn.ModuleList()
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(self.num_features, self.hidden_size, edge_dim=num_edge_features))
        
        for _ in range(self.num_layers - 2):
            self.convs.append(GATv2Conv(self.hidden_size, self.hidden_size, edge_dim=num_edge_features))
            self.batch_norms.append(BatchNorm(in_channels=self.hidden_size))
        self.convs.append(GATv2Conv(self.hidden_size, self.lin_size, edge_dim=num_edge_features))
        # # # Linear layers
        # self.linears = KAN([lin_size]*(lin_layers)+[target_size])
        self.linears = nn.ModuleList()
        for _ in range(lin_layers - 1):
            self.linears.append(nn.Linear(self.lin_size, self.lin_size))
        self.linears.append(nn.Linear(self.lin_size, self.target_size))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.bnn(x)
        edge_attr = self.bne(edge_attr)

        x = self.convs[0](x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x)
        # Apply GATConv layers with residual connections
        for i, conv in enumerate(self.convs[1:-1]):
            residual = x
            x = self.batch_norms[i](x)
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.leaky_relu(x)
            x = x + residual  # Skip connection

        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        # Apply linear layers with residual connections
        for i, linear in enumerate(self.linears[:-1]):
            residual = x
            x = linear(x)
            x = F.leaky_relu(x)
            x = x + residual  # Skip connection

        x = self.linears[-1](x)  # Final output layer
        return x
        # return self.linears(x)
    
# class PowerMLP(nn.Module):
#     def __init__(self, target_size, input_size, hidden_size=500, num_layers=1):
#         super().__init__()
        
#         self.linears = KAN([input_size]+[hidden_size]*(num_layers-2)+[target_size])
#         # self.linears = nn.ModuleList()
#         # self.linears.append(nn.Linear(in_features = input_size, out_features = hidden_size))
#         # for _ in range(num_layers - 2):
#         #     self.linears.append(nn.Linear(in_features = hidden_size, out_features = hidden_size))
#         # self.linears.append(nn.Linear(in_features = hidden_size, out_features = target_size))
        
#     def forward(self, x):
#         # x = F.leaky_relu(self.linears[0](x))
#         # for layer in self.linears[1:-1]:
#         #     res = x
#         #     x = F.leaky_relu(layer(x))
#         #     x = res+x
            
#         # x = F.leaky_relu(self.linears[-1](x))
#         return self.linears(x)
