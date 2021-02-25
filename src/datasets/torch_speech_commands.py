import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import logging
import random

new_sample_rate = 8000

logger = logging.getLogger(__name__)

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):

        if os.path.exists("../datasets/"):
            root = "../datasets/"
        elif os.path.exists("datasets/"):
            root = "datasets/"
        else:
            root = "src/datasets/"

        super().__init__(root=root, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

    def __getitem__(self, item):
        # only for debugging
        data = super().__getitem__(item)
        return data


def label_to_index(labels, word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(labels, index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def pad_mfcc(batch):
    batch = [item.squeeze().permute(1, 0) for item in batch]
    length = [item.shape[0] for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch, length

def pad_mfcc2(batch):
    batch = [item.squeeze().permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=False, padding_value=0.)
    return batch

def collate_fn(label_to_index, labels=None):
    with_mfcc = True
    classification = False
    mfcc_transf = torchaudio.transforms.MFCC(log_mels=False)

    def fn(batch):
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number

        tensors, targets, mfccs, waveforms = [], [], [], []
        n_skipped = 0

        if with_mfcc:
            if classification:
                for waveform, _, label, *_ in batch:
                    tensors += [mfcc_transf.forward(waveform)]
                    targets += [label_to_index(labels, label)]
                    waveforms += [waveform]
                tensors, length = pad_mfcc(tensors)
                targets = np.array(targets, dtype="long")
                targets_one_hot = np.zeros((targets.size, 35), dtype="long")  # TODO: 35 = num classes
                targets_one_hot[np.arange(targets.size), targets] = 1
                targets = torch.from_numpy(targets_one_hot).type(torch.LongTensor)
                length = torch.LongTensor(length)
                mfccs = None
            else:
                for waveform, _, label, *_ in batch:
                    mfcc = mfcc_transf.forward(waveform)
                    length = mfcc.shape[2]
                    target_l, target_h = int(length / 2) - 15, int(length / 2) + 15
                    tensor = torch.cat((
                        mfcc[:, :, :target_l],
                        torch.zeros((mfcc.shape[0], mfcc.shape[1], target_h - target_l)),
                        mfcc[:, :, target_h:]
                    ), dim=2)
                    target = mfcc[:, :, target_l:target_h]
                    if target.shape[2] != 30:
                        logger.error("ERROR in traget size: waveform is to small (mfcc size={}, target size={})".format(mfcc.shape, target.shape))
                        n_skipped += 1
                    else:
                        tensors += [tensor]
                        targets += [target]
                        mfccs += [mfcc]
                        waveforms += [waveform]


                # TODO: FIXME some tensors had to be skipped due to their size (Just add a random tensor twice so that the batch size is correct...)
                for _ in range(n_skipped):
                    idx = random.randint(0, len(tensors) - 1)
                    tensors += [tensors[idx]]
                    targets += [targets[idx]]
                    mfccs += [mfccs[idx]]
                    waveforms += [waveforms[idx]]

                # for self implemented transformer network
                # tensors, length = pad_mfcc(tensors)
                # targets = torch.cat(targets) # Sizes of tensors must match except in dimension 0. Got 40 and 3 in dimension 2 (The offending index is 15)
                # targets = targets.reshape(targets.shape[0], targets.shape[1]*targets.shape[2])
                # length = torch.LongTensor(length)

                # for torch transformer network
                tensors = pad_mfcc2(tensors)  #  source sqeuence length x batch size x feature number = 81x64x40
                targets = torch.cat(targets).permute(2,0,1)  #  target sequence length x batch size x feature number = 30x64x40
                length = None

            return tensors, targets, length, mfccs, waveform

        else:
            for waveform, _, label, *_ in batch:
                tensors += [waveform]
                targets += [label_to_index(labels, label)]
            tensors = pad_sequence(tensors)
            targets = torch.stack(targets)

            return tensors, targets

    return fn


def analyze_some_data(dataset):
    waveform, sample_rate, label, speaker_id, utterance_number = dataset[0]
    mfcc = torchaudio.transforms.MFCC().forward(waveform).squeeze().numpy().t()

    print("Shape of waveform: {}".format(waveform.size()))
    print("Sample rate of waveform: {}".format(sample_rate))

    fig, axs = plt.subplots(nrows=2, figsize=(20, 11))
    axs[0].plot(waveform.t().numpy())
    axs[0].set_title("Waveform")
    axs[1].imshow(mfcc, origin='lower')
    axs[1].set_title("MFCC")
    plt.show()

    labels = sorted(list(set(datapoint[2] for datapoint in dataset)))
    print(labels)


if __name__ == '__main__':
    train_set = SubsetSC("training")

    analyze_some_data(train_set)
