import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer
from graphgps.layer.gcn_conv_layer import GCNConvLayer
from graphgps.layer.unitary_conv_layer import UnitaryGCNConvLayer
from graphgps.layer.orthogonal_conv_layer import OrthogonalGCNConvLayer
from torch_geometric.data import Data


import torch.nn as nn

@register_network('custom_gnn')
class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        layers = []
        for _ in range(cfg.gnn.layers_mp):
            layers.append(conv_model(dim_in,
                                     dim_in,
                                     dropout=cfg.gnn.dropout,
                                     residual=cfg.gnn.residual))
        layers.append(ComplexToReal())
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcnconv':
            return GatedGCNLayer
        elif model_type == 'gineconv':
            return GINEConvLayer
        elif model_type == 'gcnconv':
            return GCNConvLayer
        elif model_type == 'unitarygcnconv':
            return UnitaryGCNConvLayer
        elif model_type == 'orthogonalgcnconv':
            return OrthogonalGCNConvLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch




class ComplexToReal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        if isinstance(data, Data):
            # Convert complex tensors to real tensors
            if data.x is not None and torch.is_complex(data.x):
                data.x = torch.real(data.x)
                # data.x = data.x.real.clone().detach()

            if data.edge_attr is not None and torch.is_complex(data.edge_attr):
                data.edge_attr = torch.real(data.edge_attr)
                # data_edge_attr = data.edge_attr.real.clone().detach()

            if data.pos is not None and torch.is_complex(data.pos):
                data.pos = torch.real(data.pos)
                # data.pos = data.pos.real.clone().detach()

            return data
        else:
            raise ValueError("Input must be a PyTorch Geometric Data object")