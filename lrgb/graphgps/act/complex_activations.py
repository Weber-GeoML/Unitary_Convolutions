from functools import partial

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_act


class ComplexActivation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, input):
        # Separate real and imaginary parts
        if torch.is_complex(input):
            real_part = self.activation(input.real)
            imag_part = self.activation(input.imag)
            return torch.complex(real_part, imag_part)
        else:
            return self.activation(input)



def register_complex_act(name, activation):
    complex_activation = partial(ComplexActivation, activation=activation)
    register_act(name, complex_activation)

# Register complex versions of standard activation functions
register_complex_act('c_relu', nn.ReLU())
register_complex_act('c_tanh', nn.Tanh())
register_complex_act('c_sigmoid', nn.Sigmoid())
register_complex_act('c_gelu', nn.GELU())
