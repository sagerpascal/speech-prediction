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
    # TODO: read sample rate etc. from config file
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=16000)  # , melkwargs={"hop_length": 512})
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
