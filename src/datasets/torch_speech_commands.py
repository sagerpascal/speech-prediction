import logging
import os
import random

import matplotlib.pyplot as plt
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from datasets.preprocessing import get_mfcc_transform, get_mfcc_preprocess_fn

logger = logging.getLogger(__name__)


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, conf, subset: str = None):

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

        self.mfcc_transform = get_mfcc_transform(conf)
        self.preprocess = get_mfcc_preprocess_fn(mask_pos=conf['masking']['position'],
                                                 n_frames=conf['masking']['n_frames'],
                                                 k_frames=conf['masking']['k_frames'],
                                                 start_idx=conf['masking']['start_idx'])

    def __getitem__(self, item):
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(item)

        mfcc = self.mfcc_transform(waveform)
        data, target = self.preprocess(mfcc)

        return data, target, mfcc, waveform, speaker_id, sample_rate, label, utterance_number


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
