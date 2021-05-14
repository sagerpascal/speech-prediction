import librosa
import random
import numpy as np

import time
import torch
import math
from pathlib import Path


def get_check_size(conf):
    min_size = math.ceil(
        conf['data']['transform']['hop_length'] * (conf['masking']['n_frames'] + conf['masking']['k_frames'] - 1) + conf['data']['transform']['win_length'])

    def check_size(data):
        return data.size >= min_size

    return check_size


def pitch_and_speed(conf):
    lower = conf['data']['augmentation']['pitch_and_speed']['lower']
    upper = conf['data']['augmentation']['pitch_and_speed']['upper']
    prob = conf['data']['augmentation']['pitch_and_speed']['prob']
    check_size_f = get_check_size(conf)

    def apply(data):
        if random.uniform(0, 1) <= prob:
            data_aug = data.copy()
            length_change = np.random.uniform(low=lower, high=upper)
            speed_fac = 1.0 / length_change
            tmp = np.interp(np.arange(0, len(data_aug), speed_fac), np.arange(0, len(data_aug)), data_aug)
            minlen = min(data_aug.shape[0], tmp.shape[0])
            data_aug *= 0
            data_aug[0:minlen] = tmp[0:minlen]
            if check_size_f(data_aug):
                return data_aug
            else:
                return data
        else:
            return data

    return apply


def pitch_shift(conf):
    sampling_rate = conf['data']['transform']['sample_rate']
    lower = conf['data']['augmentation']['pitch_shift']['lower']
    upper = conf['data']['augmentation']['pitch_shift']['upper']
    prob = conf['data']['augmentation']['pitch_shift']['prob']
    check_size_f = get_check_size(conf)

    def apply(data):
        if random.uniform(0, 1) <= prob:
            n_steps = random.uniform(lower, upper)
            data_aug = librosa.effects.pitch_shift(data, sampling_rate, n_steps=n_steps)
            if check_size_f(data_aug):
                return data_aug
            else:
                return data
        else:
            return data

    return apply


def time_stretch(conf):
    lower = conf['data']['augmentation']['time_stretch']['lower']
    upper = conf['data']['augmentation']['time_stretch']['upper']
    prob = conf['data']['augmentation']['time_stretch']['prob']
    check_size_f = get_check_size(conf)

    def apply(data):
        if random.uniform(0, 1) <= prob:
            rate = random.uniform(lower, upper)
            data_aug = librosa.effects.time_stretch(data, rate=rate)
            if check_size_f(data_aug):
                return data_aug
            else:
                return data
        else:
            return data

    return apply


def value_ampl(conf):
    lower = conf['data']['augmentation']['amplification']['lower']
    upper = conf['data']['augmentation']['amplification']['upper']
    prob = conf['data']['augmentation']['amplification']['prob']
    check_size_f = get_check_size(conf)

    def apply(data):
        if random.uniform(0, 1) <= prob:
            dyn_change = np.random.uniform(low=lower, high=upper)
            data_aug = data * dyn_change
            if check_size_f(data_aug):
                return data_aug
            else:
                return data
        else:
            return data

    return apply


def hpss(conf):
    prob = conf['data']['augmentation']['hpss']['prob']

    def apply(data):
        if random.uniform(0, 1) <= prob:
            return librosa.effects.hpss(data)[1]
        else:
            return data

    return apply


def augment_pipe(conf):
    pitch_and_speed_f = pitch_and_speed(conf)
    pitch_shift_f = pitch_shift(conf)
    time_stretch_f = time_stretch(conf)
    value_ampl_f = value_ampl(conf)
    hpss_f = hpss(conf)

    def pipe(data):
        data = data.squeeze()
        data = data.numpy()
        size = data.size  # size should stay the same...
        data = pitch_and_speed_f(data)
        data = time_stretch_f(data)
        data = pitch_shift_f(data)
        data = value_ampl_f(data)
        data = hpss_f(data)
        data_ = np.zeros(size, dtype=np.float)
        if data.size <= size:
            # pad with zeros
            data_[:data.size] = data
        else:
            # cut out a random sequence
            start_idx = random.randint(0, data.size-size)
            data_[:] = data[start_idx:]
        assert data_.size == size
        data = torch.as_tensor(data_, dtype=torch.float32)
        data = data.unsqueeze(0)
        return data

    return pipe


def get_augmentation(conf):
    if conf['data']['augmentation']['use_augmentation']:
        return augment_pipe(conf)
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
