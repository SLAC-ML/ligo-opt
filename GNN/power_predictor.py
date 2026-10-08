"""Power-prediction architecture and shared model construction defaults."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.conv import TransformerConv


# Shared by base training, fine-tuning, and inference. These values match the
# existing base_*_gat10_kan5.pt and fine-tuned checkpoints.
DEFAULT_MODEL_CONFIG = {
    'num_features': 4,
    'hidden_size': 1000,
    'target_size': 1,
    'num_edge_features': 2,
    'num_layers': 10,
    'lin_layers': 5,
    'lin_size': 800,
}


class LinGNN(torch.nn.Module):
    def __init__(self, num_features=4, hidden_size=700, target_size=1, num_edge_features=2, num_layers=20, lin_layers = 6, lin_size=800):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.num_layers = num_layers
        self.lin_size = lin_size
        
        self.bnn = BatchNorm(in_channels=num_features)
        self.bne = BatchNorm(in_channels=num_edge_features)
        self.batch_norms = nn.ModuleList()
        # Graph transformer layers
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(self.num_features, self.hidden_size, edge_dim=num_edge_features))
        
        for _ in range(self.num_layers - 2):
            self.convs.append(TransformerConv(self.hidden_size, self.hidden_size, edge_dim=num_edge_features))
            self.batch_norms.append(BatchNorm(in_channels=self.hidden_size))
        self.convs.append(TransformerConv(self.hidden_size, self.lin_size, edge_dim=num_edge_features))
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
        # Apply graph transformer layers with residual connections
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


def build_power_predictor(
    *, graph_layers=None, linear_layers=None, gat_layers=None, kan_layers=None,
):
    """Build the power GNN using the shared architecture defaults.

    graph_layers and linear_layers optionally override the shared depths.
    gat_layers and kan_layers are legacy keyword aliases; the implementation
    uses graph transformers followed by ordinary linear layers.
    """
    config = DEFAULT_MODEL_CONFIG.copy()
    for name, value, legacy_name, legacy_value, key in (
        ('graph_layers', graph_layers, 'gat_layers', gat_layers, 'num_layers'),
        ('linear_layers', linear_layers, 'kan_layers', kan_layers, 'lin_layers'),
    ):
        if value is not None and legacy_value is not None and value != legacy_value:
            raise ValueError(f'{name} and {legacy_name} specify different layer counts.')
        resolved = value if value is not None else legacy_value
        if resolved is not None:
            config[key] = resolved
    return LinGNN(**config)
