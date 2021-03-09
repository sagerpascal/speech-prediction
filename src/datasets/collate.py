import torch
import torchaudio
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

def pad_mfcc(batch):
    batch = [item.squeeze().permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=False, padding_value=0.)
    return batch


def normalize(tensor):
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean / tensor_minusmean.max()


def collate_fn(conf, debug=False):
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
    mask_pos = conf['masking']['position']
    n_frames = conf['masking']['n_frames']
    k_frames = conf['masking']['k_frames']

    def fn(batch):

        data, target, waveforms, mfccs = [], [], [], []
        for waveform, speaker, *_ in batch:
            mfcc = mfcc_transform.forward(waveform)
            if debug:
                mfccs.append(mfcc)
                waveforms.append(waveform)

            if mfcc.shape[2] < k_frames:
                # mfcc contains less frames than we want to predict
                data.append(torch.zeros((mfcc.shape[0], mfcc.shape[1], n_frames), dtype=torch.float))
                target.append(torch.zeros((mfcc.shape[0], mfcc.shape[1], n_frames), dtype=torch.float))
                logger.info("MFCC is smaller than frames we want to predict... ignore it")

            elif mask_pos == 'end':
                if mfcc.shape[2] <= n_frames + k_frames:
                    target_index = mfcc.shape[2] - k_frames
                    data.append(mfcc.detach().clone()[:, :, :target_index])
                    target.append(mfcc.detach().clone()[:, :, target_index:])
                    logger.info("MFCC is smaller than n_frames+k_frames...")
                else:
                    # Select a random section of the MFCC, the first part is used as data x and the second as target y
                    # use a random segment of the length n_frames + k_frames
                    random_start_index = random.randint(0, mfcc.shape[2] - (n_frames + k_frames + 1))
                    data.append(mfcc.detach().clone()[:, :, random_start_index:random_start_index+n_frames])
                    target.append(mfcc.detach().clone()[:, :, random_start_index+n_frames:random_start_index+n_frames+k_frames])

            elif mask_pos == 'beginning':
                if mfcc.shape[2] <= n_frames + k_frames:
                    target.append(mfcc.detach().clone()[:, :, :k_frames])
                    data.append(mfcc.detach().clone()[:, :, k_frames:])
                    logger.info("MFCC is smaller than n_frames+k_frames...")
                else:
                    # Select a random section of the MFCC, the first part is used as target y and the second as data x
                    # use a random segment of the length n_frames + k_frames
                    random_start_index = random.randint(0, mfcc.shape[2] - (n_frames + k_frames + 1))
                    target.append(mfcc.detach().clone()[:, :, random_start_index:random_start_index+k_frames])
                    data.append(mfcc.detach().clone()[:, :, random_start_index+k_frames:random_start_index+k_frames+n_frames])

            elif mask_pos == 'center':
                if mfcc.shape[2] <= n_frames + k_frames:
                    # Data: 0 -> n1 and n1+k_frames -> end
                    # Target: n1 -> n1+k_frames
                    n1 = (mfcc.shape[2] - k_frames) // 2
                    target.append(mfcc.detach().clone()[:, :, n1:n1+k_frames])
                    data.append(torch.cat((mfcc.detach().clone()[:, :, :n1], mfcc.detach().clone()[:, :, n1+k_frames:]), 2))
                    logger.info("MFCC is smaller than n_frames+k_frames...")
                else:
                    # Data: start_index -> start_index+n1 and start_index+n1+k_frames -> start_index+n_frames+k_frames
                    # Target: start_index+n1 -> start_index+n1+k_frames
                    random_start_index = random.randint(0, mfcc.shape[2] - (n_frames + k_frames + 1))
                    n1 = n_frames // 2
                    target.append(mfcc.detach().clone()[:, :, random_start_index+n1:random_start_index+n1+k_frames])
                    data.append(
                        torch.cat((mfcc.detach().clone()[:, :, random_start_index:random_start_index+n1],
                                   mfcc.detach().clone()[:, :, random_start_index+n1+k_frames:random_start_index+k_frames+n_frames]), 2))

            elif mask_pos == 'random':
                # mfccs = torchaudio.functional.mask_along_axis_iid(mfccs, mask_param=n_frames, mask_value=0., axis=2)
                raise NotImplementedError()

            else:
                raise AttributeError("Unknown mask_pos in config-file: {}".format(mask_pos))

        data = pad_mfcc(data)
        target = pad_mfcc(target)

        if debug:
            print("Debugging...")
            mfccs = pad_mfcc(mfccs)
            return data, target, mfccs, waveforms
        else:
            return data, target

    return fn
