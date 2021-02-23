import logging
import os
import shutil
import sys

import torch
import torchaudio
import wandb
from tqdm.auto import tqdm

from dataloader import get_loaders
from loss import get_loss
from metrics import get_metrics
from models.model import get_model
from optimizer import get_optimizer, optimizer_to
from utils.meter import AverageValueMeter

logger = logging.getLogger(__name__)


class Epoch:

    def __init__(self, model, loss, metrics, conf, stage_name, verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.conf = conf
        self.stage_name = stage_name
        self.verbose = verbose
        self._to_device()

    def _to_device(self):
        device = self.conf['device']
        self.model.to(device)
        self.loss.to(device)
        for m in self.metrics:
            m.to(device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y, input_lengths=None):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader_):
        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)  # TODO: change depending on dataset

        with tqdm(dataloader_, desc=self.stage_name, file=sys.stdout, disable=not self.verbose) as iterator:
            for x, y, length in iterator:
                x, y, length = x.to(self.conf['device']), y.to(self.conf['device']), length.to(self.conf['device']) # TODO: length only with transformer

                # x = transform(x) TODO: not with MFCC

                # train the network with one batch
                loss, y_pred = self.batch_update(x, y, length)

                # update logs: loss value
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'loss': loss_meter.mean}
                logs.update(loss_logs)

                # update logs: metrics
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, conf, optimizer, verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            conf=conf,
            stage_name='train',
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, input_lengths=None):
        self.optimizer.zero_grad()
        output, encoder_log_probs, input_lengths = self.model.forward(x, input_lengths, y) # TODO length?!
        loss = torch.nn.functional.cross_entropy(output, y) # self.loss(output, y) # TODO: replace with self.loss and add .squeeze() for M5
        loss.backward()
        self.optimizer.step()
        return loss, output


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, conf, verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            conf=conf,
            stage_name='valid',
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, input_lengths=None):
        with torch.no_grad():
            # https://datascience.stackexchange.com/questions/81727/what-would-be-the-target-input-for-transformer-decoder-during-test-phase
            output, encoder_log_probs, input_lengths  = self.model.forward(x, input_lengths, y) # TODO length?!
            loss = torch.nn.functional.cross_entropy(output, y) # TODO: add .squeeze() for M5
        return loss, output


def wandb_log_settings(conf, loader_train, loader_val):
    add_logs = {
        'size training set': len(loader_train.dataset),
        'size validation set': len(loader_val.dataset),
    }

    wandb.config.update({**conf, **add_logs})


def wandb_log_epoch(n_epoch, train_logs, valid_logs):
    logs = {
        'epoch': n_epoch,
    }
    wandb.log({**logs, **train_logs, **valid_logs})


def save_model(model, model_name, save_wandb=False):
    model_path = '../../trained_models'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    filename = 'model.pth' if save_wandb else model_name
    filename = os.path.join(model_path, filename)
    if os.path.exists(filename):
        os.remove(filename)
    torch.save(model, filename)

    if save_wandb:
        wandb.save(filename)
        shutil.copy(filename, os.path.join(model_path, '{}.pth'.format(model_name)))


def train(conf):
    loader_train, loader_val, _, n_input, n_output = get_loaders(conf)
    model = get_model(conf, n_input, n_output)
    loss = get_loss(conf)
    optimizer = get_optimizer(conf, model)
    optimizer_to(optimizer, conf['device'])
    metrics = get_metrics(conf)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=conf['scheduler_step_size'],
                                                   gamma=conf['scheduler_gamma'])

    train_epoch = TrainEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        conf=conf
    )

    valid_epoch = ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        conf=conf
    )

    best_loss = 9999999
    count_not_improved = 0

    for i in range(conf['max_number_of_epochs']):
        train_logs = train_epoch.run(loader_train)
        valid_logs = valid_epoch.run(loader_val)
        if conf['use_wandb']:
            wandb_log_epoch(i, train_logs, valid_logs)

        if valid_logs['loss'] < best_loss:
            best_loss = valid_logs['loss']
            model_name = wandb.run.name if conf['use_wandb'] else 'tsc_acf'
            save_model(model, model_name, save_wandb=conf['use_wandb'])

        else:
            count_not_improved += 1

        if i % 10 == 0:
            model_name = "{}_backup".format(wandb.run.name) if conf['use_wandb'] else 'model_backup'
            save_model(model, model_name, save_wandb=False)

        if conf['use_lr_scheduler']:
            lr_scheduler.step(epoch=i)

        if conf['early_stopping'] and count_not_improved >= 5:
            logger.info("early stopping after {} epochs".format(i))
            break
