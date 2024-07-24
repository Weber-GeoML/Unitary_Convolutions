import torch
import torch.nn as nn
from measure_smoothing import dirichlet_normalized
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, GatedGraphConv, GINConv, FiLMConv, global_mean_pool, GATConv, SuperGATConv, global_max_pool
import torch.nn.functional as F
from models.layers import TaylorGCNConv, ComplexGCNConv
# from models.test_layers import UnitaryGCNConvLayer
from models.complex_valued_layers import UnitaryGCNConvLayer
from models.real_valued_layers import OrthogonalGCNConvLayer

class RGATConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GATConv(in_features, out_features))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new

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

class GNN(torch.nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.args = args
        self.num_relations = args.num_relations
        self.layer_type = args.layer_type
        num_features = [args.input_dim] + list(args.hidden_layers) + [args.output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            layers.append(self.get_layer(in_features, out_features))
        self.layers = ModuleList(layers)
        self.dropout = Dropout(p=args.dropout)
        self.act_fn = ReLU()

        if self.args.last_layer_fa:
            if self.layer_type == "R-GCN" or self.layer_type == "GCN":
                self.last_layer_transform = torch.nn.Linear(self.args.hidden_dim, self.args.output_dim)
            elif self.layer_type == "R-GIN" or self.layer_type == "GIN":
                self.last_layer_transform = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.hidden_dim),nn.BatchNorm1d(self.args.hidden_dim), nn.ReLU(),nn.Linear(self.args.hidden_dim, self.args.output_dim))
            else:
                raise NotImplementedError

    def get_layer(self, in_features, out_features):
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "R-GAT":
            return RGATConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "R-GIN":
            return RGINConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)))
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "FiLM":
            return FiLMConv(in_features, out_features)

    def forward(self, graph, measure_dirichlet=False):
        x, edge_index, ptr, batch = graph.x, graph.edge_index, graph.ptr, graph.batch
        x = x.float()
        for i, layer in enumerate(self.layers):
            if self.layer_type in ["R-GCN", "R-GAT", "R-GIN", "FiLM"]:
                x_new = layer(x, edge_index, edge_type=graph.edge_type)
            else:
                x_new = layer(x, edge_index)
            if i != self.num_layers - 1:
                x_new = self.act_fn(x_new)
                x_new = self.dropout(x_new)
            if i == self.num_layers - 1 and self.args.last_layer_fa:
                combined_values = global_mean_pool(x, batch)
                combined_values = self.last_layer_transform(combined_values)
                if self.layer_type in ["R-GCN", "R-GIN"]:
                    x_new += combined_values[batch]
                else:
                    x_new = combined_values[batch]
            x = x_new 
        if measure_dirichlet:
            energy = dirichlet_normalized(x.cpu().numpy(), graph.edge_index.cpu().numpy())
            return energy
        x = global_mean_pool(x, batch)
        return x
    

"""
class ComplexGCN(nn.Module):
    def __init__(self, args):
        super(ComplexGCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        input_dim = args.input_dim
        hidden_dim = 128
        output_dim = args.output_dim
        num_layers = 4
        hidden_layer_dim = 128
        self.T = 20
        self.dropout = Dropout(p=args.dropout)
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            sample_layer = ComplexGCNConv(input_dim, hidden_dim)
            taylor_layer = TaylorGCNConv(sample_layer, T=self.T)
            self.conv_layers.append(taylor_layer)
            input_dim = hidden_dim
        self.hidden_layer = nn.Linear(hidden_dim, hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, output_dim)
        self.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x_real = F.relu(x.real)
            x_real = self.dropout(x_real) # added dropout
            x_imag = F.relu(x.imag)
            x_imag = self.dropout(x_imag) # added dropout
            x = torch.complex(x_real, x_imag)
        x = global_mean_pool(x.real, data.batch)  # Global pooling over nodes
        x = F.relu(self.hidden_layer(x))  # Hidden layer with ReLU activation
        x = self.output_layer(x)  # Output layer
        return x.squeeze()
    
    def reset_parameters(self):
        pass


class ComplexGCN(nn.Module):
    def __init__(self, args):
        super(ComplexGCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        input_dim = args.input_dim
        hidden_dim = 256
        output_dim = args.output_dim
        num_layers = 6
        hidden_layer_dim = 256
        self.T = 20
        self.dropout = Dropout(p=args.dropout)
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(UnitaryGCNConvLayer(input_dim, hidden_dim))
        for _ in range(num_layers):
            self.conv_layers.append(UnitaryGCNConvLayer(hidden_dim, hidden_dim, use_hermitian=True))
        self.hidden_layer = nn.Linear(hidden_dim, hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, output_dim)
        self.reset_parameters()

    def forward(self, data):
        for conv in self.conv_layers:
            data = conv(data)
        x = global_mean_pool(data.x.real, data.batch)  # Global pooling over nodes
        x = F.relu(self.hidden_layer(x))  # Hidden layer with ReLU activation
        x = self.output_layer(x)  # Output layer
        return x.squeeze()
    
    def reset_parameters(self):
        pass
"""


class UnitaryGCN(nn.Module):
    def __init__(self, args):
        super(UnitaryGCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        input_dim = args.input_dim
        hidden_dim = 256
        output_dim = args.output_dim
        num_layers = 6
        hidden_layer_dim = 256
        self.T = 20
        self.dropout = Dropout(p=args.dropout)
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(UnitaryGCNConvLayer(input_dim, hidden_dim))
        for _ in range(num_layers):
            self.conv_layers.append(UnitaryGCNConvLayer(hidden_dim, hidden_dim, use_hermitian=True))
        self.hidden_layer = nn.Linear(hidden_dim, hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, output_dim)
        self.reset_parameters()

    def forward(self, data):
        for conv in self.conv_layers:
            data = conv(data)
        x = global_mean_pool(data.x.real, data.batch)  # Global pooling over nodes
        x = F.relu(self.hidden_layer(x))  # Hidden layer with ReLU activation
        x = self.output_layer(x)  # Output layer
        return x.squeeze()
    
    def reset_parameters(self):
        pass


class OrthogonalGCN(nn.Module):
    def __init__(self, args):
        super(OrthogonalGCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        input_dim = args.input_dim
        hidden_dim = 256
        output_dim = args.output_dim
        num_layers = 6
        hidden_layer_dim = 256
        self.T = 20
        self.dropout = Dropout(p=args.dropout)
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(OrthogonalGCNConvLayer(input_dim, hidden_dim))
        for _ in range(num_layers):
            self.conv_layers.append(OrthogonalGCNConvLayer(hidden_dim, hidden_dim, use_hermitian=True))
        self.hidden_layer = nn.Linear(hidden_dim, hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, output_dim)
        self.reset_parameters()

    def forward(self, data):
        for conv in self.conv_layers:
            data = conv(data)
        x = global_mean_pool(data.x.real, data.batch)  # Global pooling over nodes
        x = F.relu(self.hidden_layer(x))  # Hidden layer with ReLU activation
        x = self.output_layer(x)  # Output layer
        return x.squeeze()
    
    def reset_parameters(self):
        pass