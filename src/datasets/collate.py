import torch
import logging

logger = logging.getLogger(__name__)


def pad_mfcc(batch):
    batch = [item.squeeze().permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=False, padding_value=0.)
    return batch


def normalize(tensor):
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean / tensor_minusmean.max()


def collate_fn(batch):
    data_b, target_b, waveform_b, mfcc_b = [], [], [], []
    for data, target, *_ in batch:
        data_b.append(data)
        target_b.append(target)

    data_b = pad_mfcc(data_b)
    target_b = pad_mfcc(target_b)

    return data_b, target_b


def collate_fn_debug(batch):
    data_b, target_b, waveform_b, mfcc_b = [], [], [], []
    for data, target, mfcc, waveform, *_ in batch:
        data_b.append(data)
        target_b.append(target)
        mfcc_b.append(mfcc)
        waveform_b.append(waveform)

    data_b = pad_mfcc(data_b)
    target_b = pad_mfcc(target_b)
    mfcc_b = pad_mfcc(mfcc_b)

    return data_b, target_b, mfcc_b, waveform_b

