import matplotlib.pyplot as plt
import sys
from tqdm.auto import tqdm
from utils.conf_reader import get_config
from dataloader import get_loaders
import numpy as np


def plot_data_examples():
    """
    Just plot a few samples from the dataset...
    """
    conf = get_config()

    loader_train, _, _, _, _ = get_loaders(conf)
    it_loader_train = iter(loader_train)
    x, y, _, original = next(it_loader_train)

    for i in range(x.shape[2]):
        data_orig = original[i].squeeze().numpy()
        data_network = x[:,i,:].squeeze().t().numpy()
        label = y[:, i, :].squeeze().t().numpy()

        min, max = np.min(data_orig), np.max(data_orig)

        fig, axs = plt.subplots(nrows=3, figsize=(8, 15))
        axs[0].imshow(data_orig, origin='lower', vmin=min, vmax=max)
        axs[0].set_title("Original Data")
        axs[1].imshow(data_network, origin='lower', vmin=min, vmax=max)
        axs[1].set_title("Input Data")
        axs[2].imshow(label, origin='lower', vmin=min, vmax=max)
        axs[2].set_title("Data to predict")
        plt.tight_layout()
        plt.show()

        # training: started with a loss of 3.6e+03


if __name__ == '__main__':
    plot_data_examples()








