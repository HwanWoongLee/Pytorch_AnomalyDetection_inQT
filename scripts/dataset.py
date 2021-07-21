from torch.utils.data import DataLoader


def create_data_loader(_dataset, _batch_size, _shuffle):
    loader = DataLoader(dataset=_dataset, batch_size=_batch_size, shuffle=_shuffle)
    return loader
