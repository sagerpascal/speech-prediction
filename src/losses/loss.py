import torch

from losses.soft_dtw import SoftDTW


class SoftDTWWrapper(SoftDTW):

    def __init__(self, conf, gamma=1., length=None):
        super().__init__(use_cuda=conf['device'] == "cuda", gamma=gamma)
        self.conf = conf
        self.bs = conf['train']['batch_size']
        self.n_features = conf['data']['transform']['n_mels'] if conf['data']['type'] == 'mel-spectro' else conf['data']['transform']['n_mfcc']
        if length is not None:
            self.length = length
        else:
            self.length = conf['masking']['k_frames']

    def __call__(self, x, y):
        # x and y should be batch_size x seq_len x dims

        assert x.shape[1] == self.length
        assert y.shape[1] == self.length
        assert x.shape[2] == self.n_features
        assert y.shape[2] == self.n_features

        return super().__call__(x, y).mean()


def get_loss(conf):
    if conf['train']['loss'] == 'mse':
        return torch.nn.MSELoss()
    elif conf['train']['loss'] == 'soft-dtw':
        return SoftDTWWrapper(conf)
    else:
        raise AttributeError("Unknown loss")
