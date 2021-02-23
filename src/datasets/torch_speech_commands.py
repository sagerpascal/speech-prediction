
from torchaudio.datasets import SPEECHCOMMANDS
import os
import matplotlib.pyplot as plt
import torch
import torchaudio
import numpy as np

new_sample_rate = 8000

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

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


def collate_fn(labels, label_to_index):

    with_mfcc = True
    mfcc_transf = torchaudio.transforms.MFCC(log_mels=False)

    def fn(batch):
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number

        tensors, targets = [], []

        if with_mfcc:
            for waveform, _, label, *_ in batch:
                tensors += [mfcc_transf.forward(waveform)]
                targets += [label_to_index(labels, label)]
            tensors, length = pad_mfcc(tensors)
            targets = np.array(targets, dtype="long")
            targets_one_hot = np.zeros((targets.size, 35), dtype="long")  # TODO: 35 = num classes
            targets_one_hot[np.arange(targets.size), targets] = 1
            targets = torch.from_numpy(targets_one_hot).type(torch.LongTensor)
            length = torch.LongTensor(length)

            return tensors, targets, length

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

    print("Shape of waveform: {}".format(waveform.size()))
    print("Sample rate of waveform: {}".format(sample_rate))

    plt.plot(waveform.t().numpy())
    plt.show()

    labels = sorted(list(set(datapoint[2] for datapoint in dataset)))
    print(labels)


if __name__ == '__main__':
    train_set = SubsetSC("training")

    analyze_some_data(train_set)



