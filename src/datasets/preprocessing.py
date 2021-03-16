import torchaudio
import torch
import logging
import random
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_mfcc_transform(conf):
    mel_spectro_args = {
        'win_length': conf['data']['transform']['win_length'],
        'hop_length': conf['data']['transform']['hop_length'],
        'n_fft': conf['data']['transform']['n_fft'],
        'f_min': conf['data']['transform']['f_min'],
        'f_max': conf['data']['transform']['f_max'],
    }

    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=conf['data']['transform']['sample_rate'],
                                                n_mfcc=conf['data']['transform']['n_mfcc'],
                                                melkwargs=mel_spectro_args,
                                                )

    return mfcc_transform


class Preprocess(nn.Module):
    """ must be a class, otherwise multiprocessing won't work """

    def __init__(self, n_frames, k_frames, start_idx):
        super(Preprocess, self).__init__()
        self.n_frames = n_frames
        self.k_frames = k_frames
        self.start_idx = start_idx

    def forward(self, mfcc):
        if mfcc.shape[2] < self.k_frames:
            # mfcc contains less frames than we want to predict -> set data and target to 0
            # data = torch.zeros((mfcc.shape[0], mfcc.shape[1], self.n_frames), dtype=torch.float)
            # target = torch.zeros((mfcc.shape[0], mfcc.shape[1], self.k_frames), dtype=torch.float)
            logger.error("MFCC is smaller than frames we want to predict... ignored")
            raise RuntimeError("MFCC too small!")
            # return data, target

        elif mfcc.shape[2] < self.n_frames + self.k_frames:
            logger.error("MFCC is smaller than n_frames+k_frames...")
            return self.forward_action_to_small(mfcc)

        else:
            if self.start_idx == 'beginning' or self.start_idx == 'sliding-window' or mfcc.shape[2] == self.n_frames + self.k_frames:
                if self.start_idx == 'sliding-window':
                    assert mfcc.shape[2] == self.n_frames + self.k_frames
                idx = 0
            elif self.start_idx == 'random':
                idx = random.randint(0, mfcc.shape[2] - (self.n_frames + self.k_frames + 1))
            else:
                raise AttributeError("Unknown value set for parameter start_idx: {}".format(self.start_idx))
            return self.forward_action(mfcc, idx)

    def forward_action_too_small(self, mfcc):
        pass

    def forward_action(self, mfcc, start_index):
        pass


class PreprocessEnd(Preprocess):

    def forward_action_too_small(self, mfcc):
        target_index = mfcc.shape[2] - self.k_frames
        data = mfcc.detach().clone()[:, :, :target_index]
        target = mfcc.detach().clone()[:, :, target_index:]
        return data, target

    def forward_action(self, mfcc, start_index):
        # Select a random section of the MFCC, the first part is used as data x and the second as target y
        # use a random segment of the length n_frames + k_frames
        data = mfcc.detach().clone()[:, :, start_index:start_index + self.n_frames]
        target = mfcc.detach().clone()[:, :, start_index + self.n_frames:start_index + self.n_frames + self.k_frames]
        return data, target


class PreprocessBeginning(Preprocess):

    def forward_action_too_small(self, mfcc):
        target = mfcc.detach().clone()[:, :, :self.k_frames]
        data = mfcc.detach().clone()[:, :, self.k_frames:]
        return data, target

    def forward_action(self, mfcc, start_index):
        # Select a random section of the MFCC, the first part is used as target y and the second as data x
        # use a random segment of the length n_frames + k_frames
        target = mfcc.detach().clone()[:, :, start_index:start_index + self.k_frames]
        data = mfcc.detach().clone()[:, :, start_index + self.k_frames:start_index + self.k_frames + self.n_frames]
        return data, target


class PreprocessCenter(Preprocess):

    def forward_action_too_small(self, mfcc):
        # Data: 0 -> n1 and n1+k_frames -> end
        # Target: n1 -> n1+k_frames
        n1 = (mfcc.shape[2] - self.k_frames) // 2
        target = mfcc.detach().clone()[:, :, n1:n1 + self.k_frames]
        data = torch.cat((mfcc.detach().clone()[:, :, :n1], mfcc.detach().clone()[:, :, n1 + self.k_frames:]), 2)
        return data, target

    def forward_action(self, mfcc, start_index):
        # Data: start_index -> start_index+n1 and start_index+n1+k_frames -> start_index+n_frames+k_frames
        # Target: start_index+n1 -> start_index+n1+k_frames
        n1 = self.n_frames // 2
        target = mfcc.detach().clone()[:, :, start_index + n1:start_index + n1 + self.k_frames]
        data = torch.cat((mfcc.detach().clone()[:, :, start_index:start_index + n1],
                          mfcc.detach().clone()[:, :,
                          start_index + n1 + self.k_frames:start_index + self.k_frames + self.n_frames]), 2)

        return data, target


def get_mfcc_preprocess_fn(mask_pos, n_frames, k_frames, start_idx):
    if mask_pos == 'beginning':
        return PreprocessBeginning(n_frames=n_frames, k_frames=k_frames, start_idx=start_idx)
    elif mask_pos == 'center':
        return PreprocessCenter(n_frames=n_frames, k_frames=k_frames, start_idx=start_idx)
    elif mask_pos == 'end':
        return PreprocessEnd(n_frames=n_frames, k_frames=k_frames, start_idx=start_idx)
    else:
        raise AttributeError("Unknown value set for parameter mask_pos: {}".format(mask_pos))
