import torch
import torchaudio
import numpy as np

def pad_mfcc(batch):
    batch = [item.squeeze().permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=False, padding_value=0.)
    return batch


def normalize(tensor):
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean / tensor_minusmean.max()


def collate_fn(conf, debug=False):
    mel_spectro_args = {
        'sample_rate': conf['data']['transform']['sample_rate'],
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
    k_frames = conf['masking']['k_frames']

    def fn(batch):

        data, target, waveforms, mfccs = [], [], [], []
        for waveform, speaker, *_ in batch:
            mfcc = mfcc_transform.forward(waveform)
            if debug:
                mfccs.append(mfcc)
                waveforms.append(waveform)

            if mask_pos == 'end':
                idx = mfcc.shape[2] - k_frames
                target.append(mfcc.detach().clone()[:, :, idx:])
                data.append(mfcc.detach().clone()[:, :, :idx])

            elif mask_pos == 'beginning':
                idx = k_frames
                target.append(mfcc.detach().clone()[:, :, :idx])
                data.append(mfcc.detach().clone()[:, :, idx:])

            elif mask_pos == 'random':
                # TODO: implement
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
