import matplotlib.pyplot as plt
import numpy as np
import os
from dataloader import get_loaders
from utils.conf_reader import get_config
from datasets.collate import collate_fn



def plot_data_examples():
    """
    Just plot a few samples from the dataset...
    """
    conf = get_config()
    conf['env']['world_size'] = 1
    conf['env']['use_data_parallel'] = False

    _, valid_loader, *_ = get_loaders(conf, device=conf['device'])
    valid_loader.collate_fn = collate_fn(conf, debug=True)

    it_loader_val = iter(valid_loader)
    data, target, mfccs, waveforms = next(it_loader_val)

    for i in range(data.shape[2]):
        waveform = waveforms[i].numpy()
        mfcc = mfccs[:, i, :].squeeze().t().numpy()
        x_sample = data[:, i, :].squeeze().t().numpy()
        y_sample = target[:, i, :].squeeze().t().numpy()

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
    os.chdir('../')
    plot_data_examples()
