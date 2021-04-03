import os

import matplotlib.pyplot as plt
import numpy as np
import librosa
from dataloader import get_loaders
from datasets.collate import collate_fn_debug
from datasets.normalization import undo_zero_norm
from utils.conf_reader import get_config


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

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        if with_waveform:
            axs[0, 0].plot(waveform.T)
            axs[0, 0].set_title("Waveform")
        axs[0, 1].imshow(complete_data, origin='lower', vmin=min, vmax=max, aspect="auto", cmap=plt.get_cmap("magma"))
        if is_mel_spectro:
            axs[0, 1].set_title("Orig. Mel-Spectrogram")
        else:
            axs[0, 1].set_title("Orig. MFCC")
        axs[1, 0].imshow(x_sample, origin='lower', vmin=min, vmax=max, aspect="auto", cmap=plt.get_cmap("magma"))
        axs[1, 0].set_title("Input Data x")
        axs[1, 1].imshow(y_sample, origin='lower', vmin=min, vmax=max, aspect="auto", cmap=plt.get_cmap("magma"))
        axs[1, 1].set_title("Label y")
        plt.tight_layout()
        plt.show()

        # save masked mfcc as png
        # plt.imshow(x_sample, origin='lower', vmin=min, vmax=max, aspect="auto")
        # plt.title("Input Data x")
        # plt.savefig("input_{}f_masked.png".format(conf['n_frames']))


if __name__ == '__main__':
    os.chdir('../')
    plot_data_examples(with_waveform=False)
