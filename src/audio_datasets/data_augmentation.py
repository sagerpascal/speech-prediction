import math
import random
import time
from pathlib import Path

import librosa
import numpy
import numpy as np
import torch


class BaseAugmentation:

    def __init__(self, conf, prob):
        self.conf = conf
        self.min_data_size = math.ceil(conf['data']['transform']['hop_length'] * (conf['masking']['n_frames'] +
                                                                                  conf['masking']['k_frames'] - 1)
                                       + conf['data']['transform']['win_length'])
        self.prob = prob

    def __call__(self, data):
        if random.uniform(0, 1) <= self.prob:
            data_aug = self.apply_augmentation(data)
            if self.check_size(data_aug):
                return data_aug
            else:
                return data
        else:
            return data

    def apply_augmentation(self, data):
        raise NotImplementedError

    def check_size(self, data):
        return data.size >= self.min_data_size


class ResampleAugmentation(BaseAugmentation):
    def __init__(self, conf):
        super().__init__(conf, prob=conf['data']['augmentation']['resample']['prob'])
        self.sampling_rate = conf['data']['transform']['sample_rate']
        self.lower = conf['data']['augmentation']['resample']['lower']
        self.upper = conf['data']['augmentation']['resample']['upper']

    def apply_augmentation(self, data):
        factor = random.uniform(self.lower, self.upper)
        data_aug = librosa.resample(data, self.sampling_rate, target_sr=int(self.sampling_rate*factor))
        return data_aug


class PitchAndSpeedAugmentation(BaseAugmentation):
    """ Resample with linear interpolation """
    def __init__(self, conf):
        super().__init__(conf, prob=conf['data']['augmentation']['pitch_and_speed']['prob'])
        self.lower = conf['data']['augmentation']['pitch_and_speed']['lower']
        self.upper = conf['data']['augmentation']['pitch_and_speed']['upper']

    def apply_augmentation(self, data):
        data_aug = data.copy()
        length_change = np.random.uniform(low=self.lower, high=self.upper)
        speed_fac = 1.0 / length_change
        tmp = np.interp(np.arange(0, len(data_aug), speed_fac), np.arange(0, len(data_aug)), data_aug)
        minlen = min(data_aug.shape[0], tmp.shape[0])
        data_aug *= 0
        data_aug[0:minlen] = tmp[0:minlen]
        return data_aug


class ValueAmplificationAugmentation(BaseAugmentation):

    def __init__(self, conf):
        super().__init__(conf, prob=conf['data']['augmentation']['amplification']['prob'])
        self.lower = conf['data']['augmentation']['amplification']['lower']
        self.upper = conf['data']['augmentation']['amplification']['upper']

    def apply_augmentation(self, data):
        dyn_change = np.random.uniform(low=self.lower, high=self.upper)
        factors = []
        for i in range(0, data.size, 1000):
            factors.append(1000*[np.random.uniform(low=self.lower, high=self.upper)])
        factors = np.array(factors).flatten()[:data.size]
        data_aug = data * factors
        return data_aug

class AugmentationPipeline:

    def __init__(self, conf):
        self.conf = conf
        self.resample_augmentation = ResampleAugmentation(conf)
        self.pitch_and_speed_augmentation = PitchAndSpeedAugmentation(conf)
        self.value_ampl_augmentation = ValueAmplificationAugmentation(conf)

    def __call__(self, data):
        data = data.squeeze()
        if not isinstance(data, numpy.ndarray):
            data = data.numpy()
        size = data.size  # size should stay the same...
        data = self.resample_augmentation(data)
        data = self.pitch_and_speed_augmentation(data)
        data = self.value_ampl_augmentation(data)
        data_ = np.zeros(size, dtype=float)
        if data.size <= size:
            # pad with zeros
            data_[:data.size] = data
        else:
            # cut out a random sequence
            start_idx = random.randint(0, data.size - size)
            data_[:] = data[start_idx:start_idx+size]
        assert data_.size == size
        data = torch.as_tensor(data_, dtype=torch.float32)
        data = data.unsqueeze(0)
        return data


def get_augmentation(conf):
    if conf['data']['augmentation']['use_augmentation']:
        return AugmentationPipeline(conf)
    else:
        return None


if __name__ == '__main__':
    # test data augmentation
    import torchaudio
    import matplotlib.pyplot as plt
    import sounddevice as sd

    conf = {'data': {'augmentation': {
        'use_augmentation': True,
        'pitch_and_speed': {
            'prob': .5,  # 1
            'lower': 0.7,  # .7
            'upper': 1.3  # 1.3
        },
        'resample': {
            'prob': .5,  # 1
            'lower': 0.7,
            'upper': 1.3
        },
        'amplification': {
            'prob': .8,  # 1
            'lower': 0.8,  # 0.8
            'upper': 1.2
        },
    }, 'transform': {'sample_rate': 16000, 'hop_length': 200, 'win_length': 400},
    },
        'masking': {'n_frames': 120, 'k_frames': 20}
    }

    augmentation = get_augmentation(conf)
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=conf['data']['transform']['sample_rate'],
                                                     win_length=400,
                                                     hop_length=200,
                                                     n_fft=512,
                                                     f_min=0,
                                                     f_max=8000,
                                                     n_mels=80,
                                                     )
    to_db = torchaudio.transforms.AmplitudeToDB()

    for fp in Path('D:/Projekte/temporal-speech-context/data/tmp/').glob('**/14-212-0000.flac'):
        waveform = torchaudio.load(str(fp))[0]

        waveform_augmented = augmentation(waveform)
        mel_spectro = to_db(transform(waveform))
        mel_spectro_augmented = to_db(transform(waveform_augmented))

        # sd.play(waveform.T, conf['data']['transform']['sample_rate'], blocking=True)
        # time.sleep(0.5)
        # sd.play(waveform_augmented.T, conf['data']['transform']['sample_rate'], blocking=True)

        fig, ax = plt.subplots(nrows=4, figsize=(20, 10))
        ax[0].plot(waveform.numpy().T)
        ax[1].plot(waveform_augmented.numpy().T)
        ax[2].imshow(mel_spectro.squeeze().numpy(), origin="lower", cmap=plt.get_cmap("magma"))
        ax[3].imshow(mel_spectro_augmented.squeeze().numpy(), origin="lower", cmap=plt.get_cmap("magma"))
        plt.tight_layout()
        plt.show()
