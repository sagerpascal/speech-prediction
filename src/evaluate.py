import platform
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from dataloader import get_loaders
from metrics import get_metrics
from models.model import get_model
from utils.log import format_logs
from utils.meter import AverageValueMeter


def calc_metrics(conf, loader_test, model, metrics):
    logs = {}
    loss_meter = AverageValueMeter()
    metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}

    with tqdm(loader_test, desc='evaluate (test set)', file=sys.stdout) as iterator:
        for x, y, _, _, _ in iterator:

            x, y = x.to(conf['device']), y.to(conf['device'])
            with torch.no_grad():
                y_pred = model.forward(x, y)
                loss = torch.nn.functional.mse_loss(y_pred, y)

                # update logs: loss value
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'loss': loss_meter.mean}
                logs.update(loss_logs)

                # update logs: metrics
                for metric_fn in metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                s = format_logs(logs)
                iterator.set_postfix_str(s)

    print("FINAL RESULTS ON TEST-SET:\n" + s)


def plot_one_predicted_batch(conf, loader_test, model):
    it_loader_test = iter(loader_test)
    x, y, _, original, waveform = next(it_loader_test)
    x, y, _, original, waveform = next(it_loader_test)
    x, y, _, original, waveform = next(it_loader_test)
    x, y, _, original, waveform = next(it_loader_test)
    x, y, _, original, waveform = next(it_loader_test)
    x, y, _, original, waveform = next(it_loader_test)
    x, y, _, original, waveform = next(it_loader_test)
    x, y, _, original, waveform = next(it_loader_test)
    x, y, _, original, waveform = next(it_loader_test)

    x_t, y_t = x.to(conf['device']), y.to(conf['device'])

    with torch.no_grad():
        y_pred = model.forward(x_t, y_t)

    i=0
    data_orig = original[i].squeeze().numpy()
    data_network = x[:, i, :].squeeze().t().numpy()
    label_gt = y[:, i, :].squeeze().t().numpy()
    label_pr = y_pred[:, i, :].squeeze().t().cpu().numpy()

    vmin, vmax = np.min(data_orig), np.max(data_orig)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    axs[0, 0].imshow(data_orig, origin='lower', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title("Original Data")
    axs[0, 1].imshow(data_network, origin='lower', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title("Input Data")
    axs[1, 0].imshow(label_gt, origin='lower', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title("Groud Truth")
    axs[1, 1].imshow(label_pr, origin='lower', vmin=vmin, vmax=vmax)
    axs[1, 1].set_title("Prediction")
    plt.tight_layout()
    plt.show()


def play_audio_files(conf, loader_test, model):
    if platform.system() == "Windows":
        import sounddevice as sd
        from librosa.feature.inverse import mfcc_to_audio
        import time
        import random
        import scipy.io.wavfile

        it_loader_test = iter(loader_test)

        # select a random batch between 1 and 10
        # for _ in range(random.randint(1, 10)):
        #     x, y, _, original, waveform = next(it_loader_test)

        x, y, _, original, waveform = next(it_loader_test)
        x, y, _, original, waveform = next(it_loader_test)
        x, y, _, original, waveform = next(it_loader_test)
        x, y, _, original, waveform = next(it_loader_test)
        x, y, _, original, waveform = next(it_loader_test)
        x, y, _, original, waveform = next(it_loader_test)
        x, y, _, original, waveform = next(it_loader_test)
        x, y, _, original, waveform = next(it_loader_test)
        x, y, _, original, waveform = next(it_loader_test)

        with torch.no_grad():
            x_t, y_t = x.to(conf['device']), y.to(conf['device'])
            y_pred = model.forward(x_t, y_t)

        # only use one example from batch -> select a random batch
        random_idx = 0  # random.randint(0, len(waveform)-1)
        while not np.all(x[:, random_idx, :].cpu().numpy()[25:54, :] == 0):
            # it works only for a fixed shape so far...
            random_idx = random.randint(0, len(waveform)-1)

        waveform = waveform[random_idx]
        original = original[random_idx].squeeze().cpu().numpy()
        x = x[:, random_idx, :].cpu().numpy()
        y_pred = y_pred[:, random_idx, :].cpu().numpy()
        y = y[:, random_idx, :].cpu().numpy()

        assert np.all(x[25:54, :] == 0)

        # Just for comparison...
        reconstructed_orig = x.copy()
        reconstructed_orig[25:55, :] = y
        reconstructed_orig = reconstructed_orig.T

        # reconstruct signal from input and prediction
        reconstructed = x.copy()
        reconstructed[25:55, :] = y_pred
        reconstructed = reconstructed.T

        print("Playing original sound...")
        time.sleep(0.5)
        sd.play(waveform.T, 16000, blocking=True)
        # scipy.io.wavfile.write('waveform.wav', 16000, waveform.T.numpy())

        print("Playing MFCC of original sound...")
        time.sleep(0.5)
        sd.play(mfcc_to_audio(original), 16000, blocking=True)
        # scipy.io.wavfile.write('MFCC.wav', 16000, mfcc_to_audio(original))

        print("Input (masked) signal...")
        time.sleep(0.5)
        sd.play(mfcc_to_audio(x.T), 16000, blocking=True)
        # scipy.io.wavfile.write('MFCC_cropped.wav', 16000, mfcc_to_audio(x.T))

        print("Playing reconstructed signal...")
        time.sleep(0.5)
        sd.play(mfcc_to_audio(reconstructed), 16000, blocking=True)
        # scipy.io.wavfile.write('MFCC_reconstructed.wav', 16000, mfcc_to_audio(reconstructed))

        print("Playing MFCC of original sound...")
        time.sleep(0.5)
        # sd.play(mfcc_to_audio(reconstructed_orig), 16000, blocking=True)


    else:
        raise AttributeError("Currently only windows supported to play audio files")


def evaluate(conf):
    if not 'load_model' in conf or conf['load_model'] == 'None':
        raise AttributeError("Load a model to run evaluation script (invalid config)")

    _, _, loader_test, n_input, n_output = get_loaders(conf)
    model = get_model(conf, n_input, n_output)
    metrics = get_metrics(conf)

    # plot_one_predicted_batch(conf, loader_test, model)
    play_audio_files(conf, loader_test, model)
    # calc_metrics(conf, loader_test, model, metrics)
