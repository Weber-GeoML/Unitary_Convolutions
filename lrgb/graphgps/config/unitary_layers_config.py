from torch_geometric.graphgym.register import register_config


@register_config('unitary_layer')
def extended_unitary_cfg(cfg):
    """Extend unitary layer config group that is first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg
    """

    # Whether to make feature layer unitary or not (default to False)
    cfg.gnn.use_hermitian = False
