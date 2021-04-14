import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import librosa
from dataloader import get_loaders
from audio_datasets.collate import collate_fn_debug
from audio_datasets.normalization import undo_zero_norm
from utils.conf_reader import get_config
import torchaudio
from audio_datasets.preprocessing import get_mfcc_transform, get_mel_spectro_transform
from sklearn.metrics import mean_squared_error
from losses.loss import SoftDTWWrapper

def plot_data_examples(with_waveform):
    """
    Just plot a few samples from the dataset...
    """
    conf = get_config()
    conf['env']['world_size'] = 1
    conf['env']['use_data_parallel'] = False
    is_mel_spectro = conf['data']['type'] = 'mel-spectro'
    mean = conf['data']['stats'][conf['data']['type']]['train']['mean']
    std = conf['data']['stats'][conf['data']['type']]['train']['std']

    _, valid_loader, *_ = get_loaders(conf, device=conf['device'], with_waveform=with_waveform)
    valid_loader.collate_fn = collate_fn_debug

    it_loader_val = iter(valid_loader)
    data, target, complete, waveforms = next(it_loader_val)

    for i in range(data.shape[2]):
        if with_waveform:
            waveform = waveforms[i].numpy()
        complete_data = complete[i, :, :].squeeze().t().numpy()
        x_sample = data[i, :, :].squeeze().t().numpy()
        y_sample = target[i, :, :].squeeze().t().numpy()

        try:
            y_sample.shape[1]
        except IndexError:
            y_sample = np.expand_dims(y_sample, axis=1)  # if only 1D

        complete_data = undo_zero_norm(complete_data, mean, std)
        x_sample = undo_zero_norm(x_sample, mean, std)
        y_sample = undo_zero_norm(y_sample, mean, std)

        if is_mel_spectro:
            complete_data = librosa.power_to_db(complete_data, ref=np.max)
            x_sample = librosa.power_to_db(x_sample, ref=np.max)
            y_sample = librosa.power_to_db(y_sample, ref=np.max)
            min, max = None, None
        else:
            min, max = np.min(complete_data), np.max(complete_data)

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
        gs1 = axs[0, 0].get_gridspec()
        gs2 = axs[1, 0].get_gridspec()
        for ax in [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]:
            ax.remove()
        axbig1 = fig.add_subplot(gs1[0, :])
        axbig2 = fig.add_subplot(gs2[1, :])
        if with_waveform:
            axbig1.plot(waveform.T)
            axbig1.set_title("Waveform")
        axbig2.imshow(complete_data, origin='lower', vmin=min, vmax=max, aspect="auto", cmap=plt.get_cmap("magma"))
        if is_mel_spectro:
            axbig2.set_title("Orig. Mel-Spectrogram")
        else:
            axbig2.set_title("Orig. MFCC")
        axs[2, 0].imshow(x_sample, origin='lower', vmin=min, vmax=max, aspect="auto", cmap=plt.get_cmap("magma"))
        axs[2, 0].set_title("Input Data x")
        axs[2, 1].imshow(y_sample, origin='lower', vmin=min, vmax=max, aspect="auto", cmap=plt.get_cmap("magma"))
        axs[2, 1].set_title("Label y")
        plt.tight_layout()
        plt.show()

        # save masked mfcc as png
        # plt.imshow(x_sample, origin='lower', vmin=min, vmax=max, aspect="auto")
        # plt.title("Input Data x")
        # plt.savefig("input_{}f_masked.png".format(conf['n_frames']))


def plot_same_text_different_speaker():
    conf = get_config()
    sdtw_1 = SoftDTWWrapper(conf, gamma=1., length=32)
    sdtw_2 = SoftDTWWrapper(conf, gamma=.1, length=32)
    is_mel_spectro = conf['data']['type'] = 'mel-spectro'
    base_fp = Path('/workspace/data_pa/TIMIT/AUDIO_FILES/ORIGINAL/train')
    data_ = []
    files = {
        base_fp / 'FCAG0' / 'SA1.WAV': 0,
        base_fp / 'FCJF0' / 'SA1.WAV': 3,
        base_fp / 'MDCM0' / 'SA1.WAV': 4,
        base_fp / 'MRDS0' / 'SA1.WAV': 0,
    }

    if is_mel_spectro:
        transform = get_mel_spectro_transform(conf).to('cpu')
    else:
        transform = get_mfcc_transform(conf).to('cpu')

    fig, axs = plt.subplots(nrows=len(files), ncols=1, figsize=(15, 10))

    for i, (file, pad) in enumerate(files.items()):
        waveform = torchaudio.load(file)
        data = transform(waveform[0])
        data = data.squeeze().numpy()
        data = data[:, pad:]
        data = data[:, :220] # reduce to 220 length
        data_.append(data)
        if is_mel_spectro:
            data = librosa.power_to_db(data, ref=np.max)
        axs[i].imshow(data, origin='lower', aspect="auto", cmap=plt.get_cmap("magma"))

    plt.tight_layout()
    plt.show()

    # Calc mse between the female and male speaker
    for (d1, d2) in [(data_[0], data_[1]), (data_[2], data_[3])]:
        print("MSE: {}".format(mean_squared_error(d1, d2)))
        stdw_1_res, stdw_2_res = [], []
        for i in range(6):
            d1_t = torch.from_numpy(np.expand_dims(d1[:, i*32:(i+1)*32], axis=0)).to('cuda')
            d2_t = torch.from_numpy(np.expand_dims(d2[:, i*32:(i+1)*32], axis=0)).to('cuda')
            d1_t = d1_t.permute(0, 2, 1)
            d2_t = d2_t.permute(0, 2, 1)
            stdw_1_res.append(sdtw_1(d1_t, d2_t).cpu().numpy())
            stdw_2_res.append(sdtw_2(d1_t, d2_t).cpu().numpy())
        print("Soft-DTW (gamma=1.): {}".format(np.mean(stdw_1_res)))
        print("Soft-DTW (gamma=.1): {}".format(np.mean(stdw_2_res)))



if __name__ == '__main__':
    os.chdir('../')
    # plot_data_examples(with_waveform=False)
    plot_same_text_different_speaker()
