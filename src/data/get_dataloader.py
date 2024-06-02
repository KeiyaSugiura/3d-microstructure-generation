from torch.utils.data import DataLoader


def get_dataloader(config, dataset):
    _C = config["dataloader"]
    return DataLoader(dataset, batch_size=config.batch_size, **_C["params"])