

  

# Unitary/Orthogonal GNN

The code here implements unitary/orthogonal graph message passing layers. Within the `layers` folder, `complex_valued_layers.py` implements complex valued layers and `real_valued_layers.py` implements layers only using real numbers (i.e. orthogonal). Both have virtually the same arguments, and we detail all of this below.

  

### Real valued layers (orthogonal)

Here, all the operations are over real numbers .  Consider an orthogonal layer imported as below.
```python
from layers.real_valued_layers import  OrthogonalGCNConvLayer
ortho_layer = OrthogonalGCNConvLayer(input_dim,
								    output_dim, 					# must be even dimensional
								    dropout  =  0.0, 				# percentage of dropout
								    residual  =  False, 			# adds residual connection after activation
								    global_bias  =  True, 			# adds a bias term after the convolution operation
								    T  =  10, 						# Truncation in the Taylor approximation
								    use_hermitian  =  False, 		# Set to True if Lie OrthoConv is desired (otherwise separable convolution is used)
								    activation  =  torch.nn.ReLU) 	# Activation applied separately to complex and real parts
```

To use this class, simply call it on a data object in Pytorch Geometric (features `x` and edges `edge_index` must be specified here by the particular dataset or input).

```python
from torch_geometric.data import Data
data = Data(x=x, edge_index=edge_index, ...)
out = ortho_layer(data)
```
Now to explain the arguments in the class, we first note that `use_hermitian` controls whether 
 to apply OrthoConv (separable form) if set to False or Lie OrthoConv if set to True. Let us cover these two cases

**OrthoConv** (`use_hermitian=False`): $A \in  \mathbb{R}^{n \times n}$ is an adjacency matrix, $X \in  \mathbb{R}^{n \times d}$ is a feature matrix, $W \in  \mathbb{R}^{d \times d'}$ and bias $b \in \mathbb{R}^{d'}$ is a feature transformation matrix and bias. The weight matrix is not constrained to be unitary (only the message passing part) in this separable form.

$ f_{Uconv}(X) = \sigma(\exp(iA) X W + b) $
The above has an imaginary number, but we perform it here only using real numbers by pairing two numbers together and treating one as imaginary and one as real (hence the need for even output dimension). Here, $\sigma$ is the chosen activation. 

**Lie OrthoConv** (`use_hermitian=True`): In this form, `input_dim` must always equal `output_dim` and we have the following form for the 

$f_{Uconv}(X) = \sigma(\exp(g_{conv}(X)) + b)$,
where
$g_{conv}(X) = A X W, \text{ where }  W + W^\top = 0.$
This procedure is fully orthogonal (not just in the node message passing part as before) as $W$ is chosen so that the operation $g_{conv}$ is in the Lie algebra of the orthogonal group. Here, $\sigma$ is the chosen activation applied separately to real and complex parts. 


**Other arguments**:
If `residual = True`, in either case, the input $X$ is added to the output. The exponential is approximated by its taylor series up to $T$ terms.
  

### Complex valued layers (unitary)

Consider a unitary layer imported as below.

```python
 from layers.complex_valued_layers import  UnitaryGCNConvLayer
 uni_layer = UnitaryGCNConvLayer(input_dim,
							    output_dim, 
							    dropout  =  0.0, 				# percentage of dropout
							    residual  =  False, 			# adds residual connection after activation
							    global_bias  =  True, 			# adds a bias term after the convolution operation
							    T  =  10, 						# Truncation in the Taylor approximation
							    use_hermitian  =  False, 		# Set to True if Lie UniConv is desired (otherwise separable convolution is used)
							    activation  =  torch.nn.ReLU) 	# Activation applied separately to complex and real parts
```
To use this class, simply call it on a data object in Pytorch Geometric.
```python
from torch_geometric.data import Data
data = Data(x=x, edge_index=edge_index, ...)
out = uni_layer(data)
```
  Now to explain the arguments in the class, we first note that `use_hermitian` controls whether 
 to apply UniConv (separable form) if set to False or Lie UniConv if set to True. Let us cover these two cases

**UniConv** (`use_hermitian=False`): $A \in  \mathbb{R}^{n \times n}$ is an adjacency matrix, $X \in  \mathbb{C}^{n \times d}$ is a feature matrix, $W \in  \mathbb{C}^{d \times d'}$ and bias $b \in \mathbb{C}^{d'}$ is a feature transformation matrix and bias. The weight matrix is not constrained to be unitary (only the message passing part) in this separable form.

$ f_{Uconv}(X) = \sigma(\exp(iA) X W + b) $
Here, $\sigma$ is the chosen activation applied separately to real and complex parts. 

**Lie UniConv** (`use_hermitian=True`): In this form, `input_dim` must always equal `output_dim` and we have the following form for the 

$f_{Uconv}(X) = \sigma(\exp(g_{conv}(X)) + b)$,
where
$g_{conv}(X) = i A X W, \text{ where }  W = W^\dagger.$
This procedure is fully unitary (not just in the node message passing part as before) as the Hermitian matrix $W$ is chosen so that the operation $g_{conv}$ is in the Lie algebra of the unitary group. Here, $\sigma$ is the chosen activation applied separately to real and complex parts. 


**Other arguments**:
If `residual = True`, in either case, the input $X$ is added to the output. The exponential is approximated by its taylor series up to $T$ terms.
  

### Python environment setup with Conda

The code is built using Pytorch Geometric. We recommend following the installation procedure for GraphGPS and LRGB (https://github.com/toenshoff/LRGB/tree/main). This is copied below.

```bash
conda  create  -n  graphgps  python=3.10

conda  activate  graphgps
conda  install  pytorch  torchvision  torchaudio  pytorch-cuda=11.7  -c  pytorch  -c  nvidia
pip  install  torch_geometric==2.3.0

pip  install  pyg_lib  torch_scatter  torch_sparse  torch_cluster  torch_spline_conv  -f  https://data.pyg.org/whl/torch-2.0.0+cu117.html

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.
conda  install  openbabel  fsspec  rdkit  -c  conda-forge
conda  install  pandas

pip  install  pytorch-lightning  yacs  torchmetrics
pip  install  performer-pytorch
pip  install  tensorboardX
pip  install  ogb
pip  install  wandb

conda  clean  --all
```
