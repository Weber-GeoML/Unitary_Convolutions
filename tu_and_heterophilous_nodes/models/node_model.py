import torch
import torch.nn as nn
import copy
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, GINConv, FiLMConv, global_mean_pool
import torch.nn.functional as F

from models.layers import TaylorGCNConv, ComplexGCNConv
from models.real_valued_layers import OrthogonalGCNConvLayer
from models.complex_valued_layers import UnitaryGCNConvLayer


class RGINConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features))))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new


class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.num_relations = args.num_relations
        self.layer_type = args.layer_type
        num_features = [args.input_dim] + list(args.hidden_layers) + [args.output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            layers.append(self.get_layer(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=args.dropout)
        self.act_fn = ReLU()

    def get_layer(self, in_features, out_features):
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)))
        elif self.layer_type == "R-GIN":
            return RGINConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "FiLM":
            return FiLMConv(in_features, out_features)
        elif self.layer_type == "Taylor":
            return TaylorGCNConv(in_features, out_features)
        elif self.layer_type == "Complex":
            return ComplexGCNConv(in_features, out_features)
        
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        for i, layer in enumerate(self.layers):
            if self.layer_type in ["R-GCN", "R-GIN"]:
                x = layer(x, edge_index, edge_type=graph.edge_type)
            else:
                x = layer(x, edge_index)
            if i != self.num_layers - 1:
                x = self.act_fn(x)
                x = self.dropout(x)
        return x


class UnitaryGCN(nn.Module):

    def __init__(self, args):
        super(UnitaryGCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.norm = torch.nn.LayerNorm(self.input_dim)
        output_dim = args.output_dim
        self.num_layers = args.num_layers
        self.T = args.T
        # self.dropout = Dropout(p=args.dropout)
        self.conv_layers.append(UnitaryGCNConvLayer(self.input_dim, self.hidden_dim, T = args.T, dropout = args.dropout))
        for _ in range(self.num_layers):
            self.conv_layers.append(UnitaryGCNConvLayer(self.hidden_dim, self.hidden_dim, use_hermitian=args.hermitian, residual = args.residual, dropout = args.dropout, T = args.T)) 
        # self.output_layer = nn.Linear(self.hidden_dim, output_dim)
        self.output_layer = SAGEConv(self.hidden_dim, output_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        pass

    def forward(self, data):
        graph = copy.deepcopy(data)
        for i, layer in enumerate(self.conv_layers):
            graph = layer(graph)
            
        return self.output_layer(graph.x.real,graph.edge_index)
    
class OrthogonalGCN(nn.Module):
    
        def __init__(self, args):
            super(OrthogonalGCN, self).__init__()
            self.conv_layers = nn.ModuleList()
            self.input_dim = args.input_dim
            self.hidden_dim = args.hidden_dim
            self.norm = torch.nn.LayerNorm(self.input_dim)
            output_dim = args.output_dim
            self.num_layers = args.num_layers
            self.T = args.T
            self.conv_layers.append(OrthogonalGCNConvLayer(self.input_dim, self.hidden_dim, T = args.T, dropout = args.dropout))
            for _ in range(self.num_layers):
                self.conv_layers.append(OrthogonalGCNConvLayer(self.hidden_dim, self.hidden_dim, use_hermitian=True, residual=True, dropout=args.dropout, T = args.T)) 
            self.output_layer = nn.Linear(self.hidden_dim, output_dim)
            self.reset_parameters()
        
        def reset_parameters(self):
            pass
    
        def forward(self, data):
            graph = copy.deepcopy(data)
            for i, layer in enumerate(self.conv_layers):
                graph = layer(graph)
                
            return self.output_layer(graph.x.real)
