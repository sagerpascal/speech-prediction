import logging
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from dataloader import get_loaders
from losses.loss import get_loss
from metrics import get_metrics
from models.model import get_model
from optimizer import get_optimizer, get_lr
from utils.ddp import setup, cleanup
from utils.log import format_logs
from utils.meter import AverageValueMeter

logger = logging.getLogger(__name__)


class Epoch:
    """ Superclass for Training and Validation Epochs """

    def __init__(self, model, loss, metrics, device, conf, stage_name, verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.device = device
        self.conf = conf
        self.stage_name = stage_name
        self.verbose = verbose
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for m in self.metrics:
            m.to(self.device)

    def batch_update(self, x, y, length, epoch):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader_, epoch_n):
        """ Run a epoch """

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader_, desc="{} (Epoch {})".format(self.stage_name, epoch_n + 1), file=sys.stdout,
                  disable=not self.verbose) as iterator:
            for x, y, length in iterator:

                if self.conf['masking']['start_idx'] == 'full':
                    _, indices = torch.sort(length, descending=True)

                    x = torch.autograd.Variable(x[indices])
                    y = torch.autograd.Variable(y[indices])
                    length = torch.autograd.Variable(length[indices])

                x, y = x.to(self.device), y.to(self.device)

                # train the network with one batch
                loss, y_pred = self.batch_update(x, y, length, epoch_n)

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
    """ A Epoch for Training """

    def __init__(self, model, loss, metrics, device, conf, optimizer, verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            device=device,
            conf=conf,
            stage_name='train',
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.use_grad_clip_norm = conf['train']['gradient_clipping']['use_grad_clip']
        self.grad_clip_threshold = conf['train']['gradient_clipping']['grad_clip_threshold']

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, length, epoch):
        self.optimizer.zero_grad()
        output = self.model.forward(x, y, seq_lengths=length, epoch=epoch)
        if isinstance(output, tuple):
            output = output[0]
        loss = self.loss(output, y)
        loss.backward()
        if self.use_grad_clip_norm:
            grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_clip_threshold)
            # for name, param in self.model.named_parameters():
            #     if param.grad.norm() >= 1.:
            #         logger.info("Gradients clipped")
            #         logger.info(name, param.grad.norm())
        self.optimizer.step()
        return loss, output


class ValidEpoch(Epoch):
    """ A Epoch for Validation """

    def __init__(self, model, loss, metrics, device, conf, verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            device=device,
            conf=conf,
            stage_name='valid',
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, length, epoch):
        with torch.no_grad():
            if self.conf['env']['use_data_parallel']:
                output = self.model.module.predict(x)
            else:
                output = self.model.predict(x, seq_lengths=length)
            if isinstance(output, tuple):
                output = output[0]
            loss = self.loss(output, y)
        return loss, output


def setup_wandb(conf):
    """ Initialize logging with wandb.ai """
    run = wandb.init(project="{}-{}".format(conf['data']['dataset'], conf['data']['type']), job_type='train')
    wandb.run.save()
    return run


def wandb_log_settings(conf, loader_train, loader_val):
    """ Log the configuration """
    add_logs = {
        'size training set': len(loader_train.dataset),
        'size validation set': len(loader_val.dataset),
    }

    wandb.config.update({**conf, **add_logs})


def wandb_log_epoch(n_epoch, lr, best_loss, train_logs, valid_logs):
    """ Log the metrics from an epoch """
    logs = {
        'epoch': n_epoch,
        'learning rate': lr,
        'smallest loss': best_loss,
    }
    for k, v in train_logs.items():
        logs[k + " train"] = v
    for k, v in valid_logs.items():
        logs[k + " valid"] = v
    wandb.log(logs)


def save_model(model, model_path, model_name, save_wandb=False):
    """ Save a model on the filesystem and on wandb.ai if available """
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if save_wandb:
        filename = 'model.pth'
        if os.path.exists(filename):
            os.remove(filename)
        torch.save(model.state_dict(), filename)
        wandb.save(filename)
        shutil.copy(filename, Path(model_path) / model_name)
    else:
        if os.path.exists(model_name):
            os.remove(model_name)
        torch.save(model.state_dict(), Path(model_path) / model_name)


def _save_logs(store, logs, mode, rank):
    for k, v in logs.items():
        store.set("{}:{}-{}".format(k, mode, rank), str(v))


def save_logs_in_store(store, rank, train_logs, valid_logs):
    """ Save the logs in a TCP-Store to synchronize across multiple distributed instances """
    _save_logs(store, train_logs, 'train', rank)
    _save_logs(store, valid_logs, 'valid', rank)


def _get_average_logs(conf, store, logs, mode):
    avg_logs = {}
    for k, v in logs.items():
        key, val = k.split(':')[0], 0.
        for rank in range(conf['env']['world_size']):
            val += float(store.get("{}:{}-{}".format(k, mode, rank)))
        avg_logs[key] = val / conf['env']['world_size']
    return avg_logs


def calculate_average_logs(conf, store, train_logs, valid_logs):
    """ Calculate the average metrics values across multiple distributed instances """
    train_logs_avg = _get_average_logs(conf, store, train_logs, 'train')
    valid_logs_avg = _get_average_logs(conf, store, valid_logs, 'valid')
    return train_logs_avg, valid_logs_avg


def train(rank=None, mport=None, store_port=None, world_size=None, conf=None):
    """ Run the training """

    is_main_process = not conf['env']['use_data_parallel'] or conf['env']['use_data_parallel'] and rank == 0

    if conf['env']['use_data_parallel']:
        torch.cuda.manual_seed_all(42)
        setup(mport, rank, world_size)
        logger.info("Running DDP on rank {}".format(rank))
        device = rank
        model = get_model(conf, device)
        store = dist.TCPStore("127.0.0.1",
                              port=store_port,
                              world_size=conf['env']['world_size'],
                              is_master=is_main_process,
                              )
        model.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    else:
        device = conf['device']
        model = get_model(conf, device)
        model.to(device)

    loader_train, loader_val, _ = get_loaders(conf, device)
    loss = get_loss(conf)
    optimizer = get_optimizer(conf, model)
    metrics = get_metrics(conf, device)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=conf['lr_scheduler']['step_size'],
                                                   gamma=conf['lr_scheduler']['gamma'])

    train_epoch = TrainEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=is_main_process,
        conf=conf
    )

    valid_epoch = ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=is_main_process,
        conf=conf
    )

    wandb_run = None
    if conf['use_wandb'] and is_main_process:
        wandb_run = setup_wandb(conf)
        wandb_log_settings(conf, loader_train, loader_val)

    best_loss = 999999999999
    count_not_improved = 0

    for i in range(conf['train']['max_number_of_epochs']):

        if conf['env']['use_data_parallel']:
            loader_train.sampler.set_epoch(i)
            loader_val.sampler.set_epoch(i)
            dist.barrier()

        train_logs = train_epoch.run(loader_train, i)
        valid_logs = valid_epoch.run(loader_val, i)

        if conf['env']['use_data_parallel']:
            save_logs_in_store(store, rank, train_logs, valid_logs)
            dist.barrier()

        if is_main_process:
            if conf['env']['use_data_parallel']:
                train_logs, valid_logs = calculate_average_logs(conf, store, train_logs, valid_logs)

            if valid_logs['loss'] < best_loss:
                best_loss = valid_logs['loss']
                model_name = wandb.run.name if conf['use_wandb'] else 'tsc_acf'
                model_name = "{}.pth".format(model_name)
                model_path = '/workspace/data_pa/trained_models'

                save_model(model, model_path, model_name, save_wandb=conf['use_wandb'])
                logger.info("Model saved (loss={})".format(best_loss))
                count_not_improved = 0

                if conf['env']['use_data_parallel']:
                    model_fp = Path(model_path) / model_name
                    store.set("model_filename", str(model_fp.resolve()))
                    store.set("model_update_flag", str(True))

            else:
                count_not_improved += 1
                if conf['env']['use_data_parallel']:
                    store.set("model_update_flag", str(False))

            if conf['use_wandb']:
                wandb_log_epoch(i, get_lr(optimizer), best_loss, train_logs, valid_logs)

        if conf['env']['use_data_parallel']:
            dist.barrier()  # Other processes have to load model saved by process 0
            if not is_main_process and bool(store.get("model_update_flag")):
                map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                filename = store.get("model_filename").decode("utf-8")
                model.load_state_dict(torch.load(filename, map_location=map_location))

        if (i + 1) % conf['train']['backup_frequency'] == 0 and is_main_process:
            model_name = "{}-backup-{}.pth".format(wandb.run.name, i + 1) if conf[
                'use_wandb'] else 'model-backup-{}.pth'.format(i + 1)
            save_model(model, '/workspace/data_pa/trained_models', model_name, save_wandb=False)
            logger.info("Model saved as backup after {} epochs".format(i))

        if is_main_process and train_logs['loss'] < 0.0001 or conf['train'][
            'early_stopping'] and count_not_improved >= 5:
            logger.info("early stopping after {} epochs".format(i))
            if conf['env']['use_data_parallel']:
                # TODO: Fixme
                raise KeyboardInterrupt
            break

        if conf['lr_scheduler']['activate']:
            lr_scheduler.step()

    if wandb_run is not None:
        wandb_run.finish()

    dist.barrier()
    del store
    cleanup()

    if is_main_process:
        time.sleep(10)  # wait for other processes to terminate
