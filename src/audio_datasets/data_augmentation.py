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


class PitchAndSpeedAugmentation(BaseAugmentation):

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


class PitchShiftAugmentation(BaseAugmentation):

    def __init__(self, conf):
        super().__init__(conf, prob=conf['data']['augmentation']['pitch_shift']['prob'])
        self.sampling_rate = conf['data']['transform']['sample_rate']
        self.lower = conf['data']['augmentation']['pitch_shift']['lower']
        self.upper = conf['data']['augmentation']['pitch_shift']['upper']

    def apply_augmentation(self, data):
        n_steps = random.uniform(self.lower, self.upper)
        data_aug = librosa.effects.pitch_shift(data, self.sampling_rate, n_steps=n_steps)
        return data_aug


class TimeStretchAugmentation(BaseAugmentation):

    def __init__(self, conf):
        super().__init__(conf, prob=conf['data']['augmentation']['time_stretch']['prob'])
        self.lower = conf['data']['augmentation']['time_stretch']['lower']
        self.upper = conf['data']['augmentation']['time_stretch']['upper']

    def apply_augmentation(self, data):
        rate = random.uniform(self.lower, self.upper)
        data_aug = librosa.effects.time_stretch(data, rate=rate)
        return data_aug


class ValueAmplificationAugmentation(BaseAugmentation):

    def __init__(self, conf):
        super().__init__(conf, prob=conf['data']['augmentation']['amplification']['prob'])
        self.lower = conf['data']['augmentation']['amplification']['lower']
        self.upper = conf['data']['augmentation']['amplification']['upper']

    def apply_augmentation(self, data):
        dyn_change = np.random.uniform(low=self.lower, high=self.upper)
        data_aug = data * dyn_change
        return data_aug


class HpssAugmentation(BaseAugmentation):

    def __init__(self, conf):
        super().__init__(conf, prob=conf['data']['augmentation']['hpss']['prob'])

    def apply_augmentation(self, data):
        return librosa.effects.hpss(data)[1]


class AugmentationPipeline:

    def __init__(self, conf):
        self.conf = conf
        self.pitch_and_speed_augmentation = PitchAndSpeedAugmentation(conf)
        self.time_stretch_augmentation = TimeStretchAugmentation(conf)
        self.pitch_shift_augmentation = PitchShiftAugmentation(conf)
        self.value_ampl_augmentation = ValueAmplificationAugmentation(conf)
        self.hpss_augmentation = HpssAugmentation(conf)

    def __call__(self, data):
        data = data.squeeze()
        if not isinstance(data, numpy.ndarray):
            data = data.numpy()
        size = data.size  # size should stay the same...
        data = self.pitch_and_speed_augmentation(data)
        data = self.time_stretch_augmentation(data)
        data = self.pitch_shift_augmentation(data)
        data = self.value_ampl_augmentation(data)
        data = self.hpss_augmentation(data)
        data_ = np.zeros(size, dtype=float)
        if data.size <= size:
            # pad with zeros
            data_[:data.size] = data
        else:
            # cut out a random sequence
            start_idx = random.randint(0, data.size - size)
            data_[:] = data[start_idx:]
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
            'prob': 1.,  # 1
            'lower': 0.7,
            'upper': 1.3
        },
        'hpss': {
            'prob': .1  # 0.1
        },
        'amplification': {
            'prob': 1.,  # 1
            'lower': 0.8,
            'upper': 1.2
        },
        'time_stretch': {
            'prob': 1.,  # 1
            'lower': 1.,  # 1.
            'upper': 1.01  # 1.01
        },
        'pitch_shift': {
            'prob': 0.,
            'lower': 0.99,
            'upper': 1.01
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

    for fp in Path('/Users/pascal/Documents/Projects/MSE/temporal-speech-context/data/').glob('**/*.WAV'):
        waveform = torchaudio.load(str(fp))[0]

        waveform_augmented = augmentation(waveform)
        mel_spectro = to_db(transform(waveform))
        mel_spectro_augmented = to_db(transform(waveform_augmented))

        sd.play(waveform.T, conf['data']['transform']['sample_rate'], blocking=True)
        time.sleep(0.5)
        sd.play(waveform_augmented.T, conf['data']['transform']['sample_rate'], blocking=True)

        fig, ax = plt.subplots(nrows=4, figsize=(20, 10))
        ax[0].plot(waveform.numpy().T)
        ax[1].plot(waveform_augmented.numpy().T)
        ax[2].imshow(mel_spectro.squeeze().numpy(), origin="lower", cmap=plt.get_cmap("magma"))
        ax[3].imshow(mel_spectro_augmented.squeeze().numpy(), origin="lower", cmap=plt.get_cmap("magma"))
        plt.tight_layout()
        plt.show()
