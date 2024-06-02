from torch import optim


def get_optimizer(config, model):
    _C = config["optimizer"]
    optimizer_name = _C["name"]
    assert hasattr(optim, optimizer_name), f"Optimizer {optimizer_name} not found in torch.optim"
    return getattr(optim, optimizer_name)(model.parameters(), **_C["params"])