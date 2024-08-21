from torch_geometric.graphgym.register import register_config


@register_config('unitary_layer')
def extended_unitary_cfg(cfg):
    """Extend unitary layer config group that is first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg
    """

    # Whether to make feature layer unitary or not (default to False)
    cfg.gnn.use_hermitian = False

    # Type of layer to use to aggregate edge features into nodes before convolution
    cfg.gnn.conv_setup_layer_type = 'gineconv'

    # Number of layers of the specified type to use in edge to node aggregation
    cfg.gnn.layers_conv_setup = 0
