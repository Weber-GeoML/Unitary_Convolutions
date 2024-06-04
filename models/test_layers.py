import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import typing
from typing import Callable, Optional, Union, Tuple, List
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn.inits import reset, glorot, zeros
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
    to_undirected,
)

import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros

from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    Size,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value



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
        return input



class ComplexDropout(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        # If input is complex, apply dropout only to the real part
        if torch.is_complex(x):
            mask = F.dropout(torch.ones_like(x.real), p=self.dropout, training=self.training)
            return x * mask
        else:
            # If input is real, apply dropout as usual
            return F.dropout(x, p=self.dropout, training=self.training)

class UnitaryGCNConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dropout = 0.0, residual = False, global_bias = True, T = 10, use_hermitian = False, **kwargs):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        if global_bias:
            self.bias = torch.nn.Parameter(torch.zeros(dim_out, dtype=torch.cfloat))
        else:
            self.register_parameter('bias', None)

        if use_hermitian:
            base_conv = HermitianGCNConv
        else:
            base_conv = ComplexGCNConv

        self.act = nn.Sequential(
            ComplexActivation(torch.nn.ReLU()),
            ComplexDropout(self.dropout),
        )
        self.model = TaylorGCNConv(base_conv(dim_in, dim_out, **kwargs), T = T)

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x, batch.edge_index)
        if self.bias is not None:
            batch.x = batch.x + self.bias
        batch.x = self.act(batch.x)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection
        
        return batch

class HermitianGCNConv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = False,
        normalize: bool = True,
        bias: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        # self.lin = Linear(in_channels, out_channels, bias=False,
        #                   weight_initializer='glorot')
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.lin.weight = torch.nn.Parameter(torch.complex(torch.zeros_like(self.lin.weight), self.lin.weight))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels, dtype=torch.cfloat))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.weight.data = block_diagonal_complex_init(self.lin.weight.data)
        self._cached_edge_index = None
        self._cached_adj_t = None
        if self.bias is not None:
            zeros(self.bias)


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None,
                apply_feature_lin: bool = True,
                return_feature_only: bool = False) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if apply_feature_lin:
            if not torch.is_complex(x):
                x = torch.complex(x, torch.zeros_like(x))
            if return_feature_only:
                return x
        
        x = x @ self.lin.weight + x @ self.lin.weight.conj().T
        if self.bias is not None:
            x = x + self.bias

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = 1j*self.propagate(edge_index, x=x, edge_weight=edge_weight)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

class ComplexGCNConv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = False,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        # self.lin = Linear(in_channels, out_channels, bias=False,
        #                   weight_initializer='glorot')
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.lin.weight.data = init.orthogonal_(self.lin.weight.data)
        self.lin.weight = torch.nn.Parameter(torch.complex(self.lin.weight, torch.zeros_like(self.lin.weight)))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels, dtype=torch.cfloat))
        else:
            self.register_parameter('bias', None)

        

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None
        if self.bias is not None:
            zeros(self.bias)


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None,
                apply_feature_lin: bool = True,
                return_feature_only: bool = False) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if apply_feature_lin:
            if not torch.is_complex(x):
                x = torch.complex(x, torch.zeros_like(x))
            x = self.lin(x)
            if self.bias is not None:
                x = x + self.bias
            if return_feature_only:
                return x

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = 1j*self.propagate(edge_index, x=x, edge_weight=edge_weight)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)
    

class ComplexGINEConv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = False,
        normalize: bool = True,
        bias: bool = True,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        # self.lin = Linear(in_channels, out_channels, bias=False,
        #                   weight_initializer='glorot')
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.lin.weight.data = init.orthogonal_(self.lin.weight.data)
        self.lin.weight = torch.nn.Parameter(torch.complex(self.lin.weight, torch.zeros_like(self.lin.weight)))

        if edge_dim is not None:
            self.edge_lin = torch.nn.Linear(edge_dim, in_channels, bias=False)
            self.edge_lin.weight.data = init.orthogonal_(self.edge_lin.weight.data)
            self.edge_lin.weight = torch.nn.Parameter(torch.complex(self.edge_lin.weight, torch.zeros_like(self.edge_lin.weight)))
        else:
            self.edge_lin = None

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels, dtype=torch.cfloat))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None
        if self.bias is not None:
            zeros(self.bias)


    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None,
                edge_weight: OptTensor = None,
                apply_feature_lin: bool = True,
                return_feature_only: bool = False) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if apply_feature_lin:
            if not torch.is_complex(x):
                x = torch.complex(x, torch.zeros_like(x))
            x = self.lin(x)
            if self.bias is not None:
                x = x + self.bias
            if return_feature_only:
                return x

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = 1j*self.propagate(edge_index, x=x, edge_weight=edge_weight, edge_attr=edge_attr)

        return out

    # def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        # return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
    def message(self, x_j: Tensor, edge_attr: Tensor, edge_weight: OptTensor) -> Tensor:
        if self.edge_lin is not None:
            edge_attr = self.edge_lin(edge_attr)
        if edge_weight is not None:
            message = (x_j + edge_attr) * edge_weight.view(-1, 1)
        else:
            message = x_j + edge_attr
        return torch.complex(F.relu(message.real), F.relu(message.imag))

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class TaylorGCNConv(MessagePassing):
    def __init__(
        self,
        conv: ComplexGCNConv,
        T: int = 16,
        return_real: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.conv = conv
        self.T = T
        self.return_real = return_real

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))

        c = 1.
        x = self.conv(x, edge_index, edge_weight,
                      apply_feature_lin = True,
                      return_feature_only = True)
        x_k = x.clone()  # Create a copy of the input tensor

        for k in range(self.T):
            x_k = self.conv(x_k, edge_index, edge_weight, apply_feature_lin = False) / (k+1)
            x += x_k
        if self.return_real:
            x = x.real
        return x
    
def block_diagonal_complex_init(weight_matrix, block_size=2, bound=0.5):
    # Get the size of the weight matrix
    n = weight_matrix.size(0)

    weight_matrix = torch.zeros_like(weight_matrix)
    
    # Calculate the number of blocks along one dimension
    num_blocks = (n + block_size - 1) // block_size
    
    for i in range(num_blocks):
        # Calculate the actual block size for the current block
        actual_block_size = min(block_size, n - i * block_size)
        
        # Initialize the real and imaginary parts separately with Gaussian noise
        real_part = torch.randn(actual_block_size, actual_block_size) * bound
        imag_part = torch.randn(actual_block_size, actual_block_size) * bound
        
        # Combine real and imaginary parts into a complex tensor
        block = torch.view_as_complex(torch.stack([real_part, imag_part], dim=-1))
        
        # Place the block on the diagonal of the weight matrix
        start_row = i * block_size
        end_row = start_row + actual_block_size
        weight_matrix[start_row:end_row, start_row:end_row] = block
    
    return weight_matrix

@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> OptPairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(  # noqa: F811
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


class GINEConv(MessagePassing):
    """
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, 'in_features'):
                in_channels = nn.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = Linear(edge_dim, in_channels)

        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()


    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)
    

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
    

class UnitaryGINEConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dropout = 0.0, residual = True, global_bias = True, T = 10, use_hermitian = False, **kwargs):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(2*dim_out, 2*dim_out),
            torch.nn.ReLU(),
            torch.nn.Linear(2*dim_out, 2*dim_out),
        )
        if global_bias:
            self.bias = torch.nn.Parameter(torch.zeros(dim_out, dtype=torch.cfloat))
        else:
            self.register_parameter('bias', None)

        if use_hermitian:
            raise NotImplementedError("Hermitian GINEConv not implemented yet")
        else:
            base_conv = ComplexGINEConv

        self.act = nn.Sequential(
            ComplexActivation(torch.nn.ReLU()),
            ComplexDropout(self.dropout),
        )
        self.model = TaylorGCNConv(base_conv(dim_in, dim_out, **kwargs), T = T)

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x, batch.edge_index)
        if self.residual:
            batch.x = x_in + batch.x
        # split real and imaginary parts
        x_real = batch.x.real
        x_imag = batch.x.imag
        # concatenate them
        x = torch.cat([x_real, x_imag], dim=-1)
        # pass through neural network
        batch.x = self.nn(x)
        
        return batch