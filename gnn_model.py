"""
Model definitions for heterogeneous cell line-drug link regression.
Feature-dominant modeling: large initial projection → GraphSAGE → low-dim embeddings.

Note: Due to PyG's to_hetero FX tracing, we cannot use LazyLinear or dynamic init
inside forward(). The projection is applied BEFORE to_hetero wrapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FeatureProjector(nn.Module):
    """Project high-dim node features to lower dim before GNN.

    This module runs BEFORE to_hetero wrapping, projecting each node type's
    features independently. Uses two-layer MLP with batch norm for better capacity.
    """

    def __init__(self, dims_dict: dict, project_dim: int = 256, dropout: float = 0.3):
        """
        Args:
            dims_dict: {node_type: input_dim}, e.g. {"cell_line": 19215, "drug": 397}
            project_dim: output dimension after projection
            dropout: dropout rate
        """
        super().__init__()
        self.project_dim = project_dim
        self.projectors = nn.ModuleDict()
        self.batch_norms = nn.ModuleDict()
        self.projectors2 = nn.ModuleDict()
        self.batch_norms2 = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)

        for node_type, in_dim in dims_dict.items():
            # Two-layer projection with batch norm
            hidden = max(project_dim * 2, 512)
            self.projectors[node_type] = nn.Linear(in_dim, hidden)
            self.batch_norms[node_type] = nn.BatchNorm1d(hidden)
            self.projectors2[node_type] = nn.Linear(hidden, project_dim)
            self.batch_norms2[node_type] = nn.BatchNorm1d(project_dim)

    def forward(self, x_dict: dict) -> dict:
        out = {}
        for node_type, x in x_dict.items():
            if node_type in self.projectors:
                h = self.projectors[node_type](x)
                h = self.batch_norms[node_type](h)
                h = F.relu(h)
                h = self.dropout(h)
                h = self.projectors2[node_type](h)
                h = self.batch_norms2[node_type](h)
                out[node_type] = F.relu(h)
            else:
                out[node_type] = x
        return out


class GNN(nn.Module):
    """GraphSAGE encoder (to be used with to_hetero).

    Architecture:
    - Three GraphSAGE layers with dropout, batch norm, and residual connections
    - Output: node embeddings of size out_channels (default 16)
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 16,
        dropout: float = 0.3,
        num_layers: int = 3,
        heads: int = 4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        from torch_geometric.nn import GATConv
        # First layer
        self.convs.append(GATConv((-1, -1), hidden_channels, heads=heads, concat=True, add_self_loops=False))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv((-1, -1), hidden_channels, heads=heads, concat=True, add_self_loops=False))
        # Output layer
        self.convs.append(GATConv((-1, -1), out_channels, heads=1, concat=False, add_self_loops=False))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x_new = conv(x, edge_index)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x_new
        # Final layer without activation (embeddings)
        x = self.convs[-1](x, edge_index)
        return x


class LinkPredictor(nn.Module):
    """Deep MLP decoder with batch norm and residual: cell+drug embeddings → IC50."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        dropout: float = 0.3,
        num_layers: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        # Input layer
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.output = nn.Linear(hidden_channels, out_channels)

    def forward(self, cell_line_emb: torch.Tensor, drug_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cell_line_emb, drug_emb], dim=-1)
        
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            identity = x if i > 0 and x.shape[-1] == layer.out_features else None
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            if identity is not None:
                x = x + identity  # Residual connection
            x = self.dropout(x)
        
        x = self.output(x)
        return x.view(-1)
