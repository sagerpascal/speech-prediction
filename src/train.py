import logging
import os
import shutil
import sys
from pathlib import Path
import torch
import wandb
from tqdm.auto import tqdm
import models
import numpy as np
from dataloader import get_loaders
from loss import get_loss
from metrics import get_metrics
from models.model import get_model
from optimizer import get_optimizer, optimizer_to
from utils.log import format_logs
from utils.meter import AverageValueMeter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils.ddp import setup, cleanup

logger = logging.getLogger(__name__)


# TODO's:
# cleanup code
# Study mask of transformers -> can they be used to mask a certain area?
# use torchaudio.functional.mask_along_axis to mask a certain area instead of own implementation


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

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader_):
        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader_, desc=self.stage_name, file=sys.stdout, disable=not self.verbose) as iterator:
            for x, y in iterator:
                x, y = x.to(self.conf['device']), y.to(self.conf['device'])

                # train the network with one batch
                loss, y_pred = self.batch_update(x, y)

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
                    s = format_logs(logs)
                    iterator.set_postfix_str(s)

        torch.cuda.empty_cache()
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

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        output = self.model.forward(x, y)
        # TODO with custom transformer model: output, encoder_log_probs, input_lengths = self.model.forward(x, input_lengths, y)
        # TODO: classification with transformer:  torch.nn.functional.cross_entropy(output, y)
        loss = torch.nn.functional.mse_loss(output, y)
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

    def batch_update(self, x, y):
        with torch.no_grad():
            # https://datascience.stackexchange.com/questions/81727/what-would-be-the-target-input-for-transformer-decoder-during-test-phase
            # TODO with custom transformer model: output, encoder_log_probs, input_lengths  = self.model.forward(x, input_lengths, y)
            output = self.model.forward(x, y)
            loss = torch.nn.functional.mse_loss(output, y)
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
    for k, v in train_logs.items():
        logs[k + " train"] = v
    for k, v in valid_logs.items():
        logs[k + " valid"] = v
    wandb.log(logs)


def save_model(model, model_path, model_name, save_wandb=False):
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if save_wandb:
        filename = 'model.pth'
        if os.path.exists(filename):
            os.remove(filename)
        torch.save(model, filename)
        wandb.save(filename)
        shutil.copy(filename, Path(model_path) / model_name)
    else:
        if os.path.exists(model_name):
            os.remove(model_name)
        torch.save(model, Path(model_path) / model_name)



def train(rank=None, world_size=None, conf=None):
    is_main_process = not conf['env']['use_data_parallel'] or conf['env']['use_data_parallel'] and rank == 0

    model = get_model(conf)

    if conf['env']['use_data_parallel']:
        torch.cuda.manual_seed_all(42)
        setup(rank, world_size)
        logger.info("Running DDP on rank {}".format(rank))
        model.to(rank)
        model = DDP(model, device_ids=[rank])
        device = rank
    else:
        device = conf['device']
        model.to(device)

    loader_train, loader_val, _ = get_loaders(conf, device)
    loss = get_loss(conf)
    optimizer = get_optimizer(conf, model)
    metrics = get_metrics(conf)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=conf['lr_scheduler']['step_size'],
                                                   gamma=conf['lr_scheduler']['gamma'])

    train_epoch = TrainEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        verbose=is_main_process,
        conf=conf
    )

    valid_epoch = ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        verbose=is_main_process,
        conf=conf
    )

    if conf['use_wandb'] and is_main_process:
        wandb_log_settings(conf, loader_train, loader_val)

    best_loss = 9999999
    count_not_improved = 0

    for i in range(conf['train']['max_number_of_epochs']):

        if conf['env']['use_data_parallel']:
            loader_train.sampler.set_epoch(i)
            loader_val.sampler.set_epoch(i)
            dist.barrier()

        train_logs = train_epoch.run(loader_train)
        valid_logs = valid_epoch.run(loader_val)
        if conf['use_wandb'] and is_main_process:
            wandb_log_epoch(i, train_logs, valid_logs)

        dist.barrier()

        if valid_logs['loss'] < best_loss:
            model_name = wandb.run.name if conf['use_wandb'] else 'tsc_acf'
            model_path = '/workspace/data_pa/trained_models'
            if is_main_process:
                # for parallel models: only save file once!
                best_loss = valid_logs['loss']

                save_model(model, model_path, model_name, save_wandb=conf['use_wandb'])
                logger.info("Model saved (loss={})".format(best_loss))
                count_not_improved = 1

            if conf['env']['use_data_parallel']:
                dist.barrier()  # Other processes have to load model saved by process 0
                map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                model.load_state_dict(torch.load(Path(model_path) / model_name, map_location=map_location))

        elif is_main_process:
            count_not_improved += 1

        if i % 50 == 0 and is_main_process:
            model_name = "{}_backup".format(wandb.run.name) if conf['use_wandb'] else 'model_backup'
            save_model(model, '/workspace/data_pa/trained_models', model_name, save_wandb=False)
            logger.info("Model saved as backup after {} epochs".format(i))

        if conf['lr_scheduler']['activate']:
            lr_scheduler.step(epoch=i)

        if train_logs['loss'] < 0.0001 or conf['train']['early_stopping'] and count_not_improved >= 5:
            logger.info("early stopping after {} epochs".format(i))
            break

    return train_logs, valid_logs
