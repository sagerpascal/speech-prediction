import numpy as np
import torch
from robust_loss_pytorch.adaptive import AdaptiveLossFunction

from losses.soft_dtw import SoftDTW


class SoftDTWWrapper(SoftDTW):

    def __init__(self, conf, gamma=1., length=None, dist_func=SoftDTW._abs_dist_func):
        super().__init__(use_cuda=conf['device'] == "cuda", gamma=gamma, normalize=True,
                         dist_func=dist_func)
        self.conf = conf
        self.bs = conf['train']['batch_size']
        self.n_features = conf['data']['transform']['n_mels'] if conf['data']['type'] == 'mel-spectro' else \
            conf['data']['transform']['n_mfcc']
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


class WeightedL1Loss(torch.nn.L1Loss):

    def __init__(self, conf):
        super(WeightedL1Loss, self).__init__(reduction='none')
        self.conf = conf
        self.batch_size = self.conf['train']['batch_size']
        self.n_dim = conf['data']['transform']['n_mfcc'] if conf['data']['type'] == 'mfcc' else \
            conf['data']['transform']['n_mels']
        self.seq_len = self.conf['masking']['k_frames']
        self.device = conf['device']
        self.weight = self.calc_weights(
            batch_size=self.batch_size,
            seq_length=self.seq_len,
            n_dim=self.n_dim,
        ).to(self.device)

    def calc_weights(self, batch_size, seq_length, n_dim):
        weight_per_time = np.linspace(2, 0.5, seq_length)
        weight_per_time_dimensions = np.repeat(weight_per_time[:, np.newaxis], n_dim, axis=1)
        weights = np.repeat(weight_per_time_dimensions[np.newaxis, :, :], batch_size, axis=0)
        return torch.as_tensor(weights, dtype=torch.float)

    def forward(self, input, target):
        loss = super().forward(input, target)
        if input.shape[0] == self.batch_size:
            loss *= self.weight
        else:
            # only for the last batch if it contains less elements than batch_size
            weights = self.calc_weights(input.shape[0], self.seq_len, self.n_dim).to(self.device)
            loss *= weights
        return torch.mean(loss)


class AdaptiveLossFunctionWrapper(AdaptiveLossFunction):

    def __init__(self, conf):
        super(AdaptiveLossFunctionWrapper, self).__init__(num_dims=conf['masking']['k_frames']*conf['data']['transform']['n_mels'], float_dtype=torch.float32, device='cuda:0')
        self.conf = conf

    def forward(self, input, target):
        return torch.mean(self.lossfun((input - target).flatten(start_dim=1)))


def get_loss(conf):
    if conf['train']['loss'] == 'mse':
        return torch.nn.MSELoss()
    elif conf['train']['loss'] == 'mae':
        return torch.nn.L1Loss()
    elif conf['train']['loss'] == 'mae-weighted':
        return WeightedL1Loss(conf)
    elif conf['train']['loss'] == 'soft-dtw-l1':
        return SoftDTWWrapper(conf, dist_func=SoftDTW._abs_dist_func)
    elif conf['train']['loss'] == 'soft-dtw-l2':
        return SoftDTWWrapper(conf, dist_func=SoftDTW._euclidean_dist_func)
    elif conf['train']['loss'] == 'adaptive-robust':
        return AdaptiveLossFunctionWrapper(conf)
    else:
        raise AttributeError("Unknown loss")
