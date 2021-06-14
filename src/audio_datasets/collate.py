import logging

import torch

logger = logging.getLogger(__name__)


def pad_seq_3d(batch):
    batch = [item.squeeze(dim=0).permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch


def pad_seq_2d(batch):
    batch = [item.squeeze(dim=0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch


def collate_fn(batch):
    data_b, target_b, length_b = [], [], []
    for data, target, complete_data, waveform, speaker, sentence, indexes, lengths in batch:
        data_b.append(data)
        target_b.append(target)
        length_b.append(lengths)

    length_b = torch.as_tensor(length_b)

    if len(data_b[0].shape) == 2:
        data_b = pad_seq_2d(data_b)
        target_b = pad_seq_2d(target_b)
    else:
        data_b = pad_seq_3d(data_b)
        target_b = pad_seq_3d(target_b)

    return data_b, target_b, length_b


def collate_fn_debug(batch):
    data_b, target_b, waveform_b, complete_data_b, speaker_b, sentence_b, indexes_b, length_b = [], [], [], [], [], [], [], []
    for data, target, complete_data, waveform, speaker, sentence, indexes, lengths in batch:
        data_b.append(data)
        target_b.append(target)
        complete_data_b.append(complete_data)
        waveform_b.append(waveform)
        speaker_b.append(speaker)
        sentence_b.append(sentence)
        indexes_b.append(indexes)
        length_b.append(lengths)

    length_b = torch.as_tensor(length_b)

    if len(data_b[0].shape) == 2:
        data_b = pad_seq_2d(data_b)
        target_b = pad_seq_2d(target_b)
        complete_data_b = pad_seq_2d(complete_data_b)
    else:
        data_b = pad_seq_3d(data_b)
        target_b = pad_seq_3d(target_b)
        complete_data_b = pad_seq_3d(complete_data_b)

    return data_b, target_b, complete_data_b, waveform_b, speaker_b, sentence_b, indexes_b, length_b
