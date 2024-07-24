import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import pandas as pd



def dihedral_permutation_matrix(n, a, b):
    """
    Generates a permutation matrix in the regular representation of the dihedral group of order n.

    Args:
        n (int): Order of the dihedral group.
        a (int): 0 or 1 corresponding to whether or not there is a reflection.
        b (int): A number from 0 to n-1 corresponding to the number of translations in the generator of the group element.

    Returns:
        torch.Tensor: A 2n x 2n permutation matrix in the regular representation of the dihedral group.
    """
    # Create a permutation matrix for the rotation part
    P_rot = torch.zeros(2 * n, 2 * n)
    for i in range(n):
        P_rot[i, (i + b) % n] = 1
        P_rot[i + n, (n + i - b) % n + n] = 1

    # Create a permutation matrix for the reflection part
    P_refl = torch.zeros(2 * n, 2 * n)
    for i in range(n):
        P_refl[i, n + i] = 1
        P_refl[i + n, i] = 1

    # Combine the rotation and reflection parts
    P = P_rot if a == 0 else P_rot @ P_refl

    return P


def dihedral_permutation_matrix_right(n, a, b):
    """
    Generates a permutation matrix in the regular representation of the dihedral group of order n. This is for group acting on right.

    Args:
        n (int): Order of the dihedral group.
        a (int): 0 or 1 corresponding to whether or not there is a reflection.
        b (int): A number from 0 to n-1 corresponding to the number of translations in the generator of the group element.

    Returns:
        torch.Tensor: A 2n x 2n permutation matrix in the regular representation of the dihedral group.
    """
    # Create a permutation matrix for the rotation part
    P_rot = torch.zeros(2 * n, 2 * n)
    for i in range(n):
        P_rot[i, (i + b) % n] = 1
        P_rot[i + n, n + (i + b) % n] = 1

    # Create a permutation matrix for the reflection part
    P_refl = torch.zeros(2 * n, 2 * n)
    for i in range(n):
        P_refl[i, n + (-i % n)] = 1
        P_refl[i + n, (-i % n)] = 1

    # Combine the rotation and reflection parts
    P = P_rot if a == 0 else P_rot @ P_refl

    return P

class DihedralGroupConv(nn.Module):
    def __init__(self, n, in_channels, out_channels, generators, bias=True):
        super(DihedralGroupConv, self).__init__()
        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_generators = len(generators)

        self.weight = nn.Parameter(torch.Tensor(self.num_generators, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.permutation_matrices = torch.stack([dihedral_permutation_matrix_right(n, a, b) for a, b in generators])
        self.register_buffer('perm_matrix', self.permutation_matrices)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(2))

        bound = 1 / (self.in_channels*self.out_channels)
        # nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            # bound = 1 / math.sqrt(fan_in)
            # nn.init.uniform_(self.bias, -bound, bound)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        batch_size = x.size(0)
        filter = torch.einsum('gio,gcd->cido',self.perm_matrix, self.weight).reshape(2*self.n*self.in_channels,2*self.n*self.out_channels)
        out = x.reshape(batch_size, -1)@filter
        return out.view(batch_size, self.out_channels, -1)
        # out = torch.einsum('bci,gck,gio->bko', x, self.weight, self.perm_matrix)
        # return out.view(batch_size, self.out_channels, -1)

class DihedralGroupConvUnitary(DihedralGroupConv):
    def __init__(self, *kargs, T=10, **kwargs):
        super(DihedralGroupConvUnitary, self).__init__(*kargs, **kwargs)
        self.T = T

    def forward(self, x):
        batch_size = x.size(0)
        filter = torch.einsum('gio,gcd->cido',self.perm_matrix, self.weight).reshape(2*self.n*self.in_channels,2*self.n*self.out_channels)
        filter = 0.5*(filter - torch.permute(filter, [1,0]))
        z = x.reshape(batch_size, -1)
        curr_z = z
        for i in range(1, self.T+1):
            curr_z = curr_z@filter /float(i)
            z = z + curr_z

        return z.reshape(batch_size, self.out_channels, -1)
    




class ToyDataset(Dataset):
    def __init__(self, n_data, n):
        self.n_data = n_data
        self.n = n
        self.data, self.targets = self._generate_data()

    def _generate_data(self):
        data = []
        targets = []

        for _ in range(self.n_data):
            vec = np.zeros(2 * self.n)
            a1, b1 = np.random.randint(0, 2), np.random.randint(0, self.n)
            a2, b2 = np.random.randint(0, 2), np.random.randint(1, self.n)
            b2 = (b1 + b2) % self.n
            vec[a1 * self.n + b1] = 1
            vec[a2 * self.n + b2] = 1

            data.append(vec.reshape(1,-1))
            targets.append(min(abs(b1 - b2), n - abs(b2 - b1) ) if a1 == a2 else 1 + min(abs((-b1 % n) - b2), n - abs(b2 - (-b1 % n)) ))
            # targets.append(abs(a1 - a2) + ((b2 - b1) % self.n))

        return np.array(data), np.array(targets)

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return data, target
    
class DihedralGroupConvNet(nn.Module):
    def __init__(self, n, conv_layers, hidden_channels, mlp_layers, mlp_width, use_unitary=False, use_residual=True):
        super(DihedralGroupConvNet, self).__init__()
        self.n = n
        self.conv_layers = conv_layers
        self.hidden_channels = hidden_channels
        self.mlp_layers = mlp_layers
        self.mlp_width = mlp_width
        self.use_residual = use_residual

        if use_unitary:
            hidden_conv = DihedralGroupConvUnitary
        else:
            hidden_conv = DihedralGroupConv

        # Define the generators for the dihedral group
        # generators = [(0, 1), (1, 0), (1,1), (n-1,1), (n-1,0), (0,0)]
        generators = [(0, 1), (1, 0), (0,0)]
        # generators = [(0, a) for a in range(n)] + [(1,a) for a in range(n)]

        # Dihedral group convolution layer
        self.dihedral_conv = DihedralGroupConv(n, in_channels=1, out_channels=hidden_channels, generators=generators)
        # self.first_linear = nn.Linear(1,hidden_channels)
        self.first_relu = nn.ReLU()

        # Convolutional layers
        convs = []
        for _ in range(conv_layers):
            conv = hidden_conv(n, hidden_channels, hidden_channels, generators)
            if use_residual:
                conv = ResidualBlock(conv)
            convs.append(conv)
            convs.append(nn.ReLU())
        self.conv_block = nn.Sequential(*convs)

        # Average pooling
        self.avg_pool = nn.AvgPool1d(2 * n)

        # MLP layers
        mlp_blocks = []
        mlp_blocks.append(nn.Flatten())
        for _ in range(mlp_layers):
            mlp_blocks.append(nn.Linear(hidden_channels, mlp_width))
            mlp_blocks.append(nn.ReLU())
            hidden_channels = mlp_width
        mlp_blocks.append(nn.Linear(mlp_width, 1))
        self.mlp = nn.Sequential(*mlp_blocks)

    def forward(self, x):
        x = self.dihedral_conv(x)
        x = self.first_relu(x)
        x = self.conv_block(x)
        x = self.avg_pool(x).squeeze()
        x = self.mlp(x)
        return x.squeeze()

class ResidualBlock(nn.Module):
    def __init__(self, block):
        super(ResidualBlock, self).__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)



def train_model(conv_layers, hidden_channels, mlp_layers, mlp_width, use_unitary, use_residual, lr, num_epochs):
    model = DihedralGroupConvNet(n, conv_layers, hidden_channels, mlp_layers, mlp_width,
                                 use_unitary=use_unitary, use_residual=use_residual)
    model.to(device)  # Move the model to the GPU
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_trajectory = {'n': n, 'conv_layers': conv_layers, 'hidden_channels': hidden_channels,
                       'mlp_layers': mlp_layers, 'mlp_width': mlp_width, 'use_unitary': use_unitary,
                       'use_residual': use_residual, 'lr': lr, 'epochs': [], 'train_loss': [], 'test_loss': []}

    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, targets in train_dataloader:
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        loss_trajectory['epochs'].append(epoch + 1)
        loss_trajectory['train_loss'].append(epoch_loss)

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for data, targets in test_dataloader:
                data = data.to(device)
                targets = targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

            test_loss /= len(test_dataloader)
            loss_trajectory['test_loss'].append(test_loss)

        model.train()

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}')

    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for data, targets in test_dataloader:
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

        avg_loss = total_loss / len(test_dataloader)
        print(f'Evaluation Loss: {avg_loss:.4f}')

    return loss_trajectory

n_data = 500
n = 200
train_dataset = ToyDataset(n_data, n)
test_dataset = ToyDataset(100, n)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=False)

dfs = []


# Shared hyperparameters
lr = 0.001
num_epochs = 500
hidden_channels = 32
mlp_layers = 1
mlp_width = 32


# Vanilla
conv_layers = 20
use_unitary = False
use_residual = False

loss_trajectory = train_model(conv_layers, hidden_channels, mlp_layers, mlp_width,
                              use_unitary, use_residual, lr, num_epochs)
df = pd.DataFrame(loss_trajectory)
df.to_csv('loss_trajectory_n200_vanilla.csv', index=False)
dfs.append(df)


# Unitary
conv_layers = 20
use_unitary = True
use_residual = False

loss_trajectory = train_model(conv_layers, hidden_channels, mlp_layers, mlp_width,
                              use_unitary, use_residual, lr, num_epochs)


df = pd.DataFrame(loss_trajectory)
df.to_csv('loss_trajectory_n200_unitary.csv', index=False)
dfs.append(df)


# Residual
conv_layers = 20
use_unitary = False
use_residual = True

loss_trajectory = train_model(conv_layers, hidden_channels, mlp_layers, mlp_width,
                              use_unitary, use_residual, lr, num_epochs)


df = pd.DataFrame(loss_trajectory)
df.to_csv('loss_trajectory_n200_residual.csv', index=False)
dfs.append(df)


# Vanilla
conv_layers = 50
use_unitary = False
use_residual = False

loss_trajectory = train_model(conv_layers, hidden_channels, mlp_layers, mlp_width,
                              use_unitary, use_residual, lr, num_epochs)
df = pd.DataFrame(loss_trajectory)
df.to_csv('loss_trajectory_n200_vanilla_50.csv', index=False)
dfs.append(df)


# Unitary
conv_layers = 50
use_unitary = True
use_residual = False

loss_trajectory = train_model(conv_layers, hidden_channels, mlp_layers, mlp_width,
                              use_unitary, use_residual, lr, num_epochs)


df = pd.DataFrame(loss_trajectory)
df.to_csv('loss_trajectory_n200_unitary_50.csv', index=False)
dfs.append(df)


# Residual
conv_layers = 50
use_unitary = False
use_residual = True

loss_trajectory = train_model(conv_layers, hidden_channels, mlp_layers, mlp_width,
                              use_unitary, use_residual, lr, num_epochs)

df = pd.DataFrame(loss_trajectory)
df.to_csv('loss_trajectory_n200_residual_50.csv', index=False)
dfs.append(df)


# Vanilla
conv_layers = 200
use_unitary = False
use_residual = False

loss_trajectory = train_model(conv_layers, hidden_channels, mlp_layers, mlp_width,
                              use_unitary, use_residual, lr, num_epochs)
df = pd.DataFrame(loss_trajectory)
df.to_csv('loss_trajectory_n200_vanilla_200.csv', index=False)
dfs.append(df)


# Unitary
conv_layers = 200
use_unitary = True
use_residual = False

loss_trajectory = train_model(conv_layers, hidden_channels, mlp_layers, mlp_width,
                              use_unitary, use_residual, lr, num_epochs)


df = pd.DataFrame(loss_trajectory)
df.to_csv('loss_trajectory_n200_unitary_200.csv', index=False)
dfs.append(df)


# Residual
conv_layers = 200
use_unitary = False
use_residual = True

loss_trajectory = train_model(conv_layers, hidden_channels, mlp_layers, mlp_width,
                              use_unitary, use_residual, lr, num_epochs)

df = pd.DataFrame(loss_trajectory)
df.to_csv('loss_trajectory_n200_residual_200.csv', index=False)
dfs.append(df)
