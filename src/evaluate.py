import platform
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from tqdm import tqdm

from dataloader import get_loaders
from metrics import get_metrics
from models.model import get_model
from utils.log import format_logs
from utils.meter import AverageValueMeter
from datasets.collate import collate_fn_debug
from datasets.normalization import undo_zero_norm


def calc_metrics(conf, loader_test, model, metrics):
    logs = {}
    loss_meter = AverageValueMeter()
    metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}

    with tqdm(loader_test, desc='evaluate (test set)', file=sys.stdout) as iterator:
        for x, y, *_ in iterator:

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
    mean, std = conf['data']['stats']['train']['std'], conf['data']['stats']['train']['mean']

    it_loader_test = iter(loader_test)
    data, target, mfccs, waveforms = next(it_loader_test)

    x_t, y_t = data.to(conf['device']), target.to(conf['device'])

    with torch.no_grad():
        y_pred = model.forward(x_t, y_t)

    for i in range(random.randint(0, len(waveforms))):
        waveform = waveforms[i]
        mfcc = mfccs[:, i, :].squeeze().t().numpy()
        data_x = data[:, i, :].squeeze().t().numpy()
        label_gt = target[:, i, :].squeeze().t().numpy()
        label_pr = y_pred[:, i, :].squeeze().t().cpu().numpy()

        mfcc = undo_zero_norm(mfcc, mean, std)
        data_x = undo_zero_norm(data_x, mean, std)
        label_gt = undo_zero_norm(label_gt, mean, std)
        label_pr = undo_zero_norm(label_pr, mean, std)

        vmin, vmax = np.min(mfcc), np.max(mfcc)

        if waveform is not None:
            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
            gs = axs[0, 0].get_gridspec()
            axs[0, 0].remove()
            axs[0, 1].remove()
            axbig = fig.add_subplot(gs[0, :])
            axbig.plot(waveform.numpy().T)
            axbig.set_title("Waveform")
            offset = 1
        else:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
            offset = 0
        axs[0 + offset, 0].imshow(mfcc, origin='lower', vmin=vmin, vmax=vmax, aspect="auto")
        axs[0 + offset, 0].set_title("Original MFCC")
        axs[0 + offset, 1].imshow(data_x, origin='lower', vmin=vmin, vmax=vmax, aspect="auto")
        axs[0 + offset, 1].set_title("Input Data")
        axs[1 + offset, 0].imshow(label_gt, origin='lower', vmin=vmin, vmax=vmax, aspect="auto")
        axs[1 + offset, 0].set_title("Groud Truth")
        axs[1 + offset, 1].imshow(label_pr, origin='lower', vmin=vmin, vmax=vmax, aspect="auto")
        axs[1 + offset, 1].set_title("Prediction")
        plt.tight_layout()
        plt.show()


def play_audio_files(conf, loader_test, model):
    mean, std = conf['data']['stats']['train']['std'], conf['data']['stats']['train']['mean']

    if platform.system() == "Windows":
        import sounddevice as sd
    from librosa.feature.inverse import mfcc_to_audio
    import time
    import scipy.io.wavfile

    it_loader_test = iter(loader_test)

    # select a random batch between 1 and 10
    for _ in range(random.randint(1, 10)):
        x, y, original, waveform = next(it_loader_test)

    # only use one example from batch -> select a random batch
    random_idx = random.randint(0, len(waveform) - 1)
    x = x[:, random_idx, :]
    y = y[:, random_idx, :]

    with torch.no_grad():
        x_t, y_t = x.unsqueeze(1).to(conf['device']), y.unsqueeze(1).to(conf['device'])
        y_pred = model.forward(x_t, y_t).squeeze()

    waveform = waveform[random_idx].numpy()
    original = original[:, random_idx, :].squeeze().cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    x = x.cpu().numpy()
    y = y.cpu().numpy()

    original = undo_zero_norm(original, mean, std)
    y_pred = undo_zero_norm(y_pred, mean, std)
    x = undo_zero_norm(x, mean, std)
    y = undo_zero_norm(y, mean, std)

    # cut away padding
    sample_end = np.min(np.argwhere(np.all(original == 0, axis=1)))
    waveform = waveform[:sample_end]
    original = original[:sample_end, :]
    x = x[:sample_end, :]

    # # Just for comparison...
    # reconstructed_orig = x.copy()
    # start_idx = np.min(np.argwhere(np.all(x == 0, axis=1)))
    # reconstructed_orig[start_idx:start_idx + y.shape[0], :] = y
    # reconstructed_orig = reconstructed_orig.T
    #
    # # reconstruct signal from input and prediction
    # reconstructed = x.copy()
    # reconstructed[start_idx:start_idx + y.shape[0], :] = y_pred
    # reconstructed = reconstructed.T

    reconstructed_orig = np.zeros((x.shape[0] + y.shape[0], x.shape[1]), dtype=np.float)
    reconstructed = np.zeros((x.shape[0] + y.shape[0], x.shape[1]), dtype=np.float)

    if conf['masking']['position'] == 'end':
        reconstructed_orig[0:x.shape[0], :] = x
        reconstructed_orig[x.shape[0]:, :] = y
        reconstructed[0:x.shape[0], :] = x
        reconstructed[x.shape[0]:, :] = y_pred
    elif conf['masking']['position'] == 'beginning':
        reconstructed_orig[0:y.shape[0], :] = y
        reconstructed_orig[y.shape[0]:, :] = x
        reconstructed[0:y.shape[0], :] = y_pred
        reconstructed[y.shape[0]:, :] = x
    else:
        raise NotImplementedError()

    reconstructed_orig = reconstructed_orig.T
    reconstructed = reconstructed.T

    if platform.system() == "Windows":
        print("Playing original sound...")
        time.sleep(0.5)
        sd.play(waveform.T, conf['data']['transform']['sample_rate'], blocking=True)

        print("Playing MFCC of original sound...")
        time.sleep(0.5)
        sd.play(mfcc_to_audio(original.T, hop_length=conf['data']['transform']['hop_length']),
                conf['data']['transform']['sample_rate'], blocking=True)

        print("Input (masked) signal...")
        time.sleep(0.5)
        sd.play(mfcc_to_audio(x.T, hop_length=conf['data']['transform']['hop_length']),
                conf['data']['transform']['sample_rate'], blocking=True)

        print("Playing reconstructed signal...")
        time.sleep(0.5)
        sd.play(mfcc_to_audio(reconstructed, hop_length=conf['data']['transform']['hop_length']),
                conf['data']['transform']['sample_rate'], blocking=True)

        print("Playing MFCC of original sound...")
        time.sleep(0.5)
        sd.play(mfcc_to_audio(reconstructed_orig, hop_length=conf['data']['transform']['hop_length']),
                conf['data']['transform']['sample_rate'], blocking=True)

    scipy.io.wavfile.write('waveform.wav', conf['data']['transform']['sample_rate'], waveform.T)
    scipy.io.wavfile.write('MFCC.wav', conf['data']['transform']['sample_rate'],
                           mfcc_to_audio(original.T, hop_length=conf['data']['transform']['hop_length']))
    scipy.io.wavfile.write('MFCC_masked.wav', conf['data']['transform']['sample_rate'],
                           mfcc_to_audio(x.T, hop_length=conf['data']['transform']['hop_length']))
    scipy.io.wavfile.write('MFCC_reconstructed.wav', conf['data']['transform']['sample_rate'],
                           mfcc_to_audio(reconstructed, hop_length=conf['data']['transform']['hop_length']))
    scipy.io.wavfile.write('MFCC_reconstructed_orig.wav', conf['data']['transform']['sample_rate'],
                           mfcc_to_audio(reconstructed_orig, hop_length=conf['data']['transform']['hop_length']))


def evaluate(conf):
    if 'load_weights' not in conf or conf['load_weights'] == 'None':
        raise AttributeError("Load a model to run evaluation script (invalid config)")

    conf['env']['world_size'] = 1
    conf['env']['use_data_parallel'] = False
    _, _, loader_test = get_loaders(conf, device=conf['device'], with_waveform=True)
    loader_test.collate_fn = collate_fn_debug
    model = get_model(conf, conf['device'])
    metrics = get_metrics(conf, conf['device'])

    # plot_one_predicted_batch(conf, loader_test, model)
    play_audio_files(conf, loader_test, model)
    # calc_metrics(conf, loader_test, model, metrics)
