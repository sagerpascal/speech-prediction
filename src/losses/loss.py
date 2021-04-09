import torch

from losses.soft_dtw import SoftDTW


class SoftDTWWrapper(SoftDTW):

    def __init__(self, conf):
        super().__init__(use_cuda=conf['device'] == "cuda")

    def __call__(self, x, y):
        return super().__call__(x, y).mean()


def get_loss(conf):
    if conf['train']['loss'] == 'mse':
        return torch.nn.MSELoss()
    elif conf['train']['loss'] == 'soft-dtw':
        return SoftDTWWrapper(conf)
    else:
        raise AttributeError("Unknown loss")
