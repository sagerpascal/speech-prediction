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

def calc_baseline(conf, compare_model=False):
    mse_mean = []  # mse for different k if the average of x is predicted
    mse_last = []  # mse for different k, if always the last frame of x is predicted
    mse_model = [] # mse for the prediction of the model
    range_k = list(range(1, 31))
    for k_frames in range_k:
        conf['masking']['k_frames'] = k_frames
        conf['masking']['window_shift'] = conf['masking']['n_frames'] + k_frames
        conf['load_weights'] = 'exp21-n=60 k={} s=90'.format(k_frames)
        logs = {}
        loss_meters = {'{}'.format(l): AverageValueMeter() for l in ['mean', 'last']}
        metrics = get_metrics(conf, conf['device'])
        metrics_meters = {"{} mean".format(metric.__name__): AverageValueMeter() for metric in metrics}
        metrics_meters.update({"{} last".format(metric.__name__): AverageValueMeter() for metric in metrics})

        _, _, loader_test = get_loaders(conf, device=conf['device'], with_waveform=False)
        loader_test.collate_fn = collate_fn_debug
        metrics = get_metrics(conf, conf['device'])

        if compare_model:
            model = get_model(conf, conf['device'])
            metrics_meters.update({"{} model".format(metric.__name__): AverageValueMeter() for metric in metrics})
            loss_meters['model'] = AverageValueMeter()

        with tqdm(loader_test, desc='baseline (test set)', file=sys.stdout) as iterator:
            for x, y, *_ in iterator:
                x, y = x.to(conf['device']), y.to(conf['device'])

                if compare_model:
                    with torch.no_grad():
                        y_pred_model = model.predict(x)
                y_pred_mean = torch.mean(x, dim=1, keepdim=True).repeat(1, k_frames, 1)  # dim 1 is the time
                y_pred_last = x[:, -1:, :].repeat(1, k_frames, 1)

                assert y_pred_last.shape == (y.shape[0], k_frames, y.shape[2])
                assert y_pred_mean.shape == (y.shape[0], k_frames, y.shape[2])

                assert torch.equal(y_pred_last[0, 0, 0], x[0, -1, 0])
                assert torch.isclose(y_pred_mean[0, 0, 0], torch.mean(x[0:1, :, 0:1], dim=1, keepdim=True))

                predictions = {'mean': y_pred_mean, 'last': y_pred_last}
                if compare_model:
                    predictions['model'] = y_pred_model
                for pred_type, pred in predictions.items():

                    loss = torch.nn.functional.mse_loss(pred, y)

                    # update logs: loss value
                    loss_value = loss.cpu().detach().numpy()
                    loss_meters[pred_type].add(loss_value)
                    loss_logs = {'loss {}'.format(pred_type): loss_meters[pred_type].mean}
                    logs.update(loss_logs)

                    # update logs: metrics
                    for metric_fn in metrics:
                        metric_value = metric_fn(pred, y).cpu().detach().numpy()
                        metrics_meters["{} {}".format(metric_fn.__name__, pred_type)].add(metric_value)
                    metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                    logs.update(metrics_logs)

                    s = format_logs(logs)
                    iterator.set_postfix_str(s)

            mse_mean.append(logs['loss mean'])
            mse_last.append(logs['loss last'])
            if 'loss model' in logs:
                mse_model.append(logs['loss model'])

    print(mse_mean)
    print(mse_last)  # better for small k (k<10)
    print(mse_model)

    if compare_model:
        plt.plot(range_k, mse_mean, 'bs-', range_k, mse_last, 'g^-', range_k, mse_model, 'ro-')
        plt.legend(['Mean of x', 'Last value of x', 'Model Prediction'])
    else:
        plt.plot(range_k, mse_mean, 'bs-', range_k, mse_last, 'g^-')
        plt.legend(['Mean of x', 'Last value of x'])
    plt.title('MSE Baseline Predictions (n={}, s={})'.format(conf['masking']['n_frames'], conf['masking']['window_shift']))
    plt.xlabel('Number of masked frames k')
    plt.ylabel('MSE')
    plt.grid()
    plt.tight_layout()
    plt.show()

    if compare_model:
        model_mean = np.array(mse_mean) - np.array(mse_model)
        model_last = np.array(mse_last) - np.array(mse_model)
        plt.plot(range_k, model_mean, 'bs-', range_k, model_last, 'g^-', range_k, np.zeros(len(range_k)), 'ro-')
        plt.legend(['Mean minus model pred.', 'Last value minus model pred.', 'Model pred.'])
        plt.title('MSE Model vs. Baseline Comparison')
        plt.xlabel('Number of masked frames k')
        plt.ylabel('MSE')
        plt.grid()
        plt.tight_layout()
        plt.show()



def calc_metrics(conf, loader_test, model, metrics):
    logs = {}
    loss_meter = AverageValueMeter()
    metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}

    with tqdm(loader_test, desc='evaluate (test set)', file=sys.stdout) as iterator:
        for x, y, *_ in iterator:

            x, y = x.to(conf['device']), y.to(conf['device'])
            with torch.no_grad():
                y_pred = model.predict(x)  # model.forward(x, y)
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

    print("FINAL RESULTS ON TEST-SET (n={}, k={}, s={}):\n{}".format(conf['masking']['n_frames'],
                                                                     conf['masking']['k_frames'],
                                                                     conf['masking']['window_shift'],
                                                                     s))


def plot_one_predicted_batch(conf, loader_test, model):
    mean, std = conf['data']['stats']['train']['mean'], conf['data']['stats']['train']['std']

    it_loader_test = iter(loader_test)
    data, target, mfccs, waveforms = next(it_loader_test)

    x_t, y_t = data.to(conf['device']), target.to(conf['device'])

    with torch.no_grad():
        y_pred = model.predict(x_t)

    i = random.randint(0, len(waveforms))
    waveform = waveforms[i]
    mfcc = mfccs[i, :, :].squeeze().t().numpy()
    data_x = data[i, :, :].squeeze().t().numpy()
    label_gt = target[i, :, :].squeeze().t().numpy()
    label_pr = y_pred[i, :, :].squeeze().t().cpu().numpy()

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
    mean, std = conf['data']['stats']['train']['mean'], conf['data']['stats']['train']['std']

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
    random_idx = random.randint(0, x.shape[1] - 1)
    x = x[random_idx, :, :]
    y = y[random_idx, :, :]

    with torch.no_grad():
        x_t, y_t = x.unsqueeze(1).to(conf['device']), y.unsqueeze(1).to(conf['device'])
        y_pred = model.forward(x_t, y_t).squeeze()

    if waveform[random_idx] is not None:
        waveform = waveform[random_idx].numpy()
    else:
        waveform = None
    original = original[random_idx, :, :].squeeze().cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    x = x.cpu().numpy()
    y = y.cpu().numpy()

    original = undo_zero_norm(original, mean, std)
    x = undo_zero_norm(x, mean, std)
    y = undo_zero_norm(y, mean, std)
    y_pred = undo_zero_norm(y_pred, mean, std)

    # cut away padding
    if np.any(np.all(original == 0, axis=1)):
        sample_end = np.min(np.argwhere(np.all(original == 0, axis=1)))
        original = original[:sample_end, :]
        x = x[:sample_end, :]
        if waveform is not None:
            waveform = waveform[:sample_end]

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
        if waveform is not None:
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

    if waveform is not None:
        scipy.io.wavfile.write('eval_out/waveform.wav', conf['data']['transform']['sample_rate'], waveform.T)
    scipy.io.wavfile.write('eval_out/MFCC.wav', conf['data']['transform']['sample_rate'],
                           mfcc_to_audio(original.T, hop_length=conf['data']['transform']['hop_length']))
    scipy.io.wavfile.write('eval_out/MFCC_masked.wav', conf['data']['transform']['sample_rate'],
                           mfcc_to_audio(x.T, hop_length=conf['data']['transform']['hop_length']))
    scipy.io.wavfile.write('eval_out/MFCC_reconstructed.wav', conf['data']['transform']['sample_rate'],
                           mfcc_to_audio(reconstructed, hop_length=conf['data']['transform']['hop_length']))
    scipy.io.wavfile.write('eval_out/MFCC_reconstructed_orig.wav', conf['data']['transform']['sample_rate'],
                           mfcc_to_audio(reconstructed_orig, hop_length=conf['data']['transform']['hop_length']))


def evaluate(conf):
    if 'load_weights' not in conf or conf['load_weights'] == 'None':
        raise AttributeError("Load a model to run evaluation script (invalid config)")

    conf['env']['world_size'] = 1
    conf['env']['use_data_parallel'] = False
    _, _, loader_test = get_loaders(conf, device=conf['device'], with_waveform=False)
    loader_test.collate_fn = collate_fn_debug
    model = get_model(conf, conf['device'])
    metrics = get_metrics(conf, conf['device'])

    calc_baseline(conf, compare_model=True)

    # plot_one_predicted_batch(conf, loader_test, model)
    # play_audio_files(conf, loader_test, model)
    calc_metrics(conf, loader_test, model, metrics)
