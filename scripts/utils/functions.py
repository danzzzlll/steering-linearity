import torch


def make_data_steering(data, layers: int):
    return torch.cat([pos['pos'][layers] for pos in data]), torch.cat([neg['neg'][layers] for neg in data])