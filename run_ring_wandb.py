import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, GELU
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, DeepGCNLayer, SSGConv, GPSConv
from layers.complex_valued_layers import UnitaryGCNConvLayer, GroupSort
from layers.real_valued_layers import OrthogonalGCNConvLayer
import torch_geometric.transforms as Transforms
import time
import argparse
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd



# transform = Transforms.AddRandomWalkPE(walk_length=20, attr_name='pe')
transform = Transforms.AddLaplacianEigenvectorPE(k=10, attr_name='pe')

class RingGraphDataset(torch.utils.data.Dataset):
    def __init__(self, n_data=100, n_nodes=20):
        self.n_data = n_data
        self.n_nodes = n_nodes

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        # Generate a ring graph
        edge_index = torch.cat([
            torch.tensor([[i, (i + 1) % self.n_nodes] for i in range(self.n_nodes)]),
            torch.tensor([[(i + 1) % self.n_nodes, i] for i in range(self.n_nodes)])], dim=0).t().contiguous()

        # Add trivial edge features (all set to 0)
        edge_attr = torch.zeros(edge_index.size(1), 1)

        x = torch.zeros(self.n_nodes, 1)

        # Randomly select two nodes and set their features to 1
        src = torch.randint(0, self.n_nodes, (1,))
        tgt = (src + torch.randint(1, self.n_nodes, (1,))) % self.n_nodes
        x[src] = 1
        x[tgt] = 1

        # Calculate the shortest path distance between the two selected nodes
        dist = min(abs(src - tgt), self.n_nodes - abs(src - tgt)) - self.n_nodes//4

        dataset = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        dataset = transform(dataset)
        return dataset, torch.tensor(dist) 

class ComplexGCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        hidden_layer_dim: int,
        T: int = 15
    ):
        super(ComplexGCN, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        taylor_layer = UnitaryGCNConvLayer(input_dim, hidden_dim, T=T, global_bias = False, bias = False, use_hermitian=False)
        self.conv_layers.append(taylor_layer)
        for _ in range(num_layers-1):
            taylor_layer = UnitaryGCNConvLayer(hidden_dim, hidden_dim, T=T, global_bias = False, bias = False, use_hermitian=False, activation = GroupSort)
            self.conv_layers.append(taylor_layer)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.conv_layers:
            data = conv(data)
        x = global_mean_pool(data.x.real, data.batch)  # Global pooling over nodes
        x = F.gelu(self.hidden_layer(x))  # Hidden layer with GELU activation
        x = self.output_layer(x)  # Output layer
        return x.squeeze()
    
        

class FlexibleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, hidden_layer_dim):
        super(FlexibleGCN, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.hidden_layer = nn.Linear(hidden_dim, hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.gelu(x)
            x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, data.batch)  # Global pooling over nodes
        x = F.gelu(self.hidden_layer(x))  # Hidden layer with GELU activation
        x = self.output_layer(x)  # Output layer
        return x.squeeze()
    
class GraphCON(nn.Module):
    def __init__(self, GNNs, dt=1., alpha=1., gamma=1., dropout=None):
        super(GraphCON, self).__init__()
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma
        self.GNNs = GNNs  # list of the individual GNN layers
        self.dropout = dropout

    def forward(self, X0, Y0, edge_index):
        # set initial values of ODEs
        X = X0
        Y = Y0
        # solve ODEs using simple IMEX scheme
        for gnn in self.GNNs:
            Y = Y + self.dt * (torch.relu(gnn(X, edge_index)) - self.alpha * Y - self.gamma * X)
            X = X + self.dt * Y

            if (self.dropout is not None):
                Y = F.dropout(Y, self.dropout, training=self.training)
                X = F.dropout(X, self.dropout, training=self.training)

        return X, Y

class full_GraphCON_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, hidden_layer_dim, dt=1., alpha=1., gamma=1., dropout=None):
        super(full_GraphCON_GNN, self).__init__()
        self.enc = nn.Linear(input_dim, hidden_dim)
        self.GNNs = nn.ModuleList()
        for _ in range(num_layers):
            self.GNNs.append(GCNConv(hidden_dim, hidden_dim))
        self.graphcon = GraphCON(self.GNNs, dt, alpha, gamma, dropout)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_layer_dim)
        self.dec = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # compute initial values of ODEs (encode input)
        X0 = self.enc(x)
        Y0 = X0
        # stack GNNs using GraphCON
        x, _ = self.graphcon(X0, Y0, edge_index)
        # decode X state of GraphCON at final time for output nodes
        x = global_mean_pool(x, data.batch)  # Global pooling over nodes
        x = F.gelu(self.hidden_layer(x))  # Hidden layer with GELU activation
        output = self.dec(x)
        return output

class FlexibleGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, hidden_layer_dim, heads=1):
        super(FlexibleGAT, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(GATConv(input_dim, hidden_dim, heads=heads))
            input_dim = hidden_dim * heads  # Update input_dim for next layer
        self.hidden_layer = nn.Linear(hidden_dim * heads, hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.gelu(x)
            x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, data.batch)  # Global pooling over nodes
        x = F.gelu(self.hidden_layer(x))  # Hidden layer with GELU activation
        x = self.output_layer(x)  # Output layer
        return x.squeeze()
    


class OrthogonalGCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        hidden_layer_dim: int,
        T: int = 8
    ):
        super(OrthogonalGCN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.conv_layers = nn.ModuleList()
        taylor_layer = OrthogonalGCNConvLayer(input_dim, hidden_dim, T=T, global_bias = False, bias = False, activation = GroupSort, use_hermitian=False)
        self.conv_layers.append(taylor_layer)
        for _ in range(num_layers-1):
            taylor_layer = OrthogonalGCNConvLayer(hidden_dim, hidden_dim, T=T, global_bias = False, bias = False, activation = GroupSort, use_hermitian=True)
            self.conv_layers.append(taylor_layer)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, output_dim)
        self.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.conv_layers:
            data = conv(data)
        x = global_mean_pool(data.x, data.batch)  # Global pooling over nodes
        x = F.gelu(self.hidden_layer(x))  # Hidden layer with GELU activation
        x = self.output_layer(x)  # Output layer
        return x.squeeze()
    
    def reset_parameters(self):
        pass


class ResidualGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, hidden_layer_dim):
        super(ResidualGCN, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv_layer = DeepGCNLayer(
                conv=GCNConv(input_dim, hidden_dim),
                act=F.gelu,
                block='res+',
                dropout=0.5
            )
            self.conv_layers.append(conv_layer)
            input_dim = hidden_dim
        self.hidden_layer = nn.Linear(hidden_dim, hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.conv_layers:
            x = conv(x, edge_index)
        x = global_mean_pool(x, data.batch)  # Global pooling over nodes
        x = F.gelu(self.hidden_layer(x))  # Hidden layer with GELU activation
        x = self.output_layer(x)  # Output layer
        return x.squeeze()

class SSGGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, hidden_layer_dim, alpha=0.1, K=1):
        super(SSGGCN, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(SSGConv(input_dim, hidden_dim, alpha=alpha, K=K))
            input_dim = hidden_dim
        self.hidden_layer = nn.Linear(hidden_dim, hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.gelu(x)
            x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, data.batch)  # Global pooling over nodes
        x = F.gelu(self.hidden_layer(x))  # Hidden layer with GELU activation
        x = self.output_layer(x)  # Output layer
        return x.squeeze()


class GPSnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, hidden_layer_dim,heads = 4):
        super(GPSnet, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        for _ in range(num_layers):
            conv_layer = GPSConv(
                channels=hidden_dim,
                conv=GCNConv(hidden_dim, hidden_dim),
                heads=heads,
                attn_type='multihead',
                # norm = None,
            )
            self.conv_layers.append(conv_layer)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.cat([x,data.pe], dim=-1)
        x = self.embedding(x)
        for conv in self.conv_layers:
            x = conv(x, edge_index, batch=data.batch)
            x = F.gelu(x)
        x = global_mean_pool(x, data.batch)  # Global pooling over nodes
        x = F.gelu(self.hidden_layer(x))  # Hidden layer with GELU activation
        x = self.output_layer(x)  # Output layer
        return x.squeeze()




# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train GNN models on ring graph dataset')
parser.add_argument('--n_data', type=int, default=1000, help='Number of data samples')
parser.add_argument('--n_nodes', type=int, default=100, help='Number of nodes in the graph')
parser.add_argument('--input_dim', type=int, default=1, help='Input dimension')
parser.add_argument('--hidden_dim', type=int, default= 128, help='Hidden dimension for hyperparameter tuning')
parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for hyperparameter tuning')
parser.add_argument('--epochs', type=int, default=250, help='Number of epochs')
parser.add_argument('--num_layers', type=int, default=10, help='Number of convolutional layers')
parser.add_argument('--hidden_layer_dim', type=int, default=128, help='Dimension of the hidden layer')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., "cuda:0")')
parser.add_argument('--model', type=str, default='Unitary GCN', help='Model choice')
parser.add_argument('--run_name', type=str, default='ring_gnn_updated', help='Run name for saving csv')
args = parser.parse_args()

# Initialize wandb
wandb.init(project="ring_gnn")

# Log hyperparameters
config = wandb.config
config.n_data = args.n_data
config.n_nodes = args.n_nodes
config.input_dim = args.input_dim
config.hidden_dim = args.hidden_dim
config.output_dim = args.output_dim
config.lr = args.lr
config.epochs = args.epochs
config.num_layers = args.num_layers
config.hidden_layer_dim = args.hidden_layer_dim
config.batch_size = args.batch_size
config.device = args.device
config.model = args.model

# Create dataset and dataloader
dataset = RingGraphDataset(config.n_data, config.n_nodes)
loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, pin_memory_device = args.device)

# Create test dataset and dataloader
test_dataset = RingGraphDataset(500, config.n_nodes)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True, pin_memory_device = args.device)

# Create validation dataset and dataloader
val_dataset = RingGraphDataset(200, config.n_nodes)
val_loader = DataLoader(val_dataset, batch_size=200, shuffle=False, pin_memory=True, pin_memory_device = args.device)

# Initialize models, optimizers, and loss
def get_model(model):
    if model == 'Unitary GCN': 
        return ComplexGCN(config.input_dim, config.hidden_dim, config.output_dim, config.num_layers, config.hidden_layer_dim)
    elif model == 'Vanilla GCN':
        return FlexibleGCN(config.input_dim, config.hidden_dim, config.output_dim, config.num_layers, config.hidden_layer_dim)
    elif model == 'Graph Attention Network':
        return FlexibleGAT(config.input_dim, config.hidden_dim, config.output_dim, config.num_layers, config.hidden_layer_dim, heads=8)
    elif model == 'Residual GCN':
        return ResidualGCN(config.input_dim, config.hidden_dim, config.output_dim, config.num_layers, config.hidden_layer_dim)
    elif model == 'Spectral Convolution':
        return SSGGCN(config.input_dim, config.hidden_dim, config.output_dim, config.num_layers, config.hidden_layer_dim, alpha=0.1, K=5)
    elif model == 'GPS':
        return GPSnet(config.input_dim+10, config.hidden_dim, config.output_dim, config.num_layers, config.hidden_layer_dim, heads=8)
    elif model == 'GraphCON GCN':
        return full_GraphCON_GNN(config.input_dim, config.hidden_dim, config.output_dim, config.num_layers, config.hidden_layer_dim)
    elif model == 'Orthogonal GCN':
        return OrthogonalGCN(config.input_dim, config.hidden_dim, config.output_dim, config.num_layers, config.hidden_layer_dim)
    else:
        raise ValueError(f'Invalid model: {model}')
    


# Training loop
# Log model name
model_name = config.model
wandb.run.name = config.model
model = get_model(config.model)


print(f'Training {config.model}')
model.to(config.device)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, min_lr=0.000001)
criterion = nn.L1Loss()

# Dictionary to store train and test losses for each model
losses = {'train': [], 'test': []}

for epoch in range(config.epochs):
    total_loss = 0
    start_time = time.time()
    for data, target in loader:
        optimizer.zero_grad()
        target = target.to(config.device)
        output = model(data.to(config.device))
        loss = criterion(output.squeeze(), target.squeeze().float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses['train'].append(total_loss / len(loader))

    # Calculate test error
    model.eval()
    test_loss = 0
    val_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target.to(config.device)
            output = model(data.to(config.device))
            test_loss += criterion(output.squeeze(), target.squeeze().float()).item()
        for data, target in val_loader:
            target = target.to(config.device)
            output = model(data.to(config.device))
            val_loss += criterion(output.squeeze(), target.squeeze().float()).item()
    losses['test'].append(test_loss / len(test_loader))

    scheduler.step(val_loss)

    # Log train and test losses
    wandb.log({"Train Loss": total_loss / len(loader), "Test Loss": test_loss / len(test_loader), "Epoch": epoch})

    model.train()

    # Log time taken for each epoch
    end_time = time.time()
    epoch_time = end_time - start_time
    wandb.log({"Epoch Time": epoch_time, "Epoch": epoch})

    print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(loader)}, Test Loss: {test_loss / len(test_loader)}")


# Save results to csv
import random

losses['epoch'] = list(range(config.epochs))

# Combine `args` and `losses` into a single dictionary
out_dict = {**vars(args), **losses}
print(out_dict)

# Convert the combined dictionary to a DataFrame
df = pd.DataFrame.from_dict(out_dict)

# Generate a random number of length 10
random_number = ''.join(random.choices('0123456789', k=10))

# Log final results
wandb.log({"Final Train Loss": total_loss / len(loader), "Final Test Loss": test_loss / len(test_loader)})

# Save the DataFrame to a CSV file with the random number added to the file name
# df.to_csv(f'./results/ring/{args.run_name}_{config.model}_{config.n_nodes}_{random_number}.csv')

