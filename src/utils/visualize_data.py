import matplotlib.pyplot as plt
import numpy as np

from dataloader import get_loaders
from utils.conf_reader import get_config


def plot_data_examples():
    """
    Just plot a few samples from the dataset...
    """
    conf = get_config()

    _, valid_loader, *_ = get_loaders(conf)
    it_loader_val = iter(valid_loader)
    x, y, mfccs, waveforms = next(it_loader_val)

    for i in range(x.shape[2]):
        waveform = waveforms[i].numpy()
        mfcc = mfccs[:, i, :].squeeze().t().numpy()
        x_sample = x[:, i, :].squeeze().t().numpy()
        y_sample = y[:, i, :].squeeze().t().numpy()

        try:
            y_sample.shape[1]
        except IndexError:
            y_sample = np.expand_dims(y_sample, axis=1)  # if only 1D

        min, max = np.min(mfcc), np.max(mfcc)

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        axs[0, 0].plot(waveform.T)
        axs[0, 0].set_title("Waveform")
        axs[0, 1].imshow(mfcc, origin='lower', vmin=min, vmax=max, aspect="auto")
        axs[0, 1].set_title("MFCC")
        axs[1, 0].imshow(x_sample, origin='lower', vmin=min, vmax=max, aspect="auto")
        axs[1, 0].set_title("Input Data x")
        axs[1, 1].imshow(y_sample, origin='lower', vmin=min, vmax=max, aspect="auto")
        axs[1, 1].set_title("Label y")
        plt.tight_layout()
        plt.show()

        # save masked mfcc as png
        # plt.imshow(x_sample, origin='lower', vmin=min, vmax=max, aspect="auto")
        # plt.title("Input Data x")
        # plt.savefig("input_{}f_masked.png".format(conf['n_frames']))


if __name__ == '__main__':
    plot_data_examples()
