import logging

import torch

logger = logging.getLogger(__name__)


def pad_mfcc_3d(batch):
    batch = [item.squeeze(dim=0).permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch


def pad_mfcc_2d(batch):
    batch = [item.squeeze(dim=0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch


def normalize(tensor):
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean / tensor_minusmean.max()


def collate_fn(batch):
    data_b, target_b = [], []
    for data, target, *_ in batch:
        data_b.append(data)
        target_b.append(target)

    if len(data_b[0].shape) == 2:
        data_b = pad_mfcc_2d(data_b)
        target_b = pad_mfcc_2d(target_b)
    else:
        data_b = pad_mfcc_3d(data_b)
        target_b = pad_mfcc_3d(target_b)

    return data_b, target_b


def collate_fn_debug(batch):
    data_b, target_b, waveform_b, mfcc_b, speaker_b, sentence_b, indexes_b = [], [], [], [], [], [], []
    for data, target, mfcc, waveform, speaker, sentence, indexes in batch:
        data_b.append(data)
        target_b.append(target)
        mfcc_b.append(mfcc)
        waveform_b.append(waveform)
        speaker_b.append(speaker)
        sentence_b.append(sentence)
        indexes_b.append(indexes)

    if len(data_b[0].shape) == 2:
        data_b = pad_mfcc_2d(data_b)
        target_b = pad_mfcc_2d(target_b)
        mfcc_b = pad_mfcc_2d(mfcc_b)
    else:
        data_b = pad_mfcc_3d(data_b)
        target_b = pad_mfcc_3d(target_b)
        mfcc_b = pad_mfcc_3d(mfcc_b)

    return data_b, target_b, mfcc_b, waveform_b, speaker_b, sentence_b, indexes_b
