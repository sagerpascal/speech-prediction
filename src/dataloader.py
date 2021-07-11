import logging
import platform

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from audio_datasets.collate import collate_fn
from audio_datasets.data_augmentation import get_augmentation
from audio_datasets.dataset import AudioDataset
from audio_datasets.dataset_h5 import AudioDatasetH5
from audio_datasets.torch_speech_commands import SubsetSC

logger = logging.getLogger(__name__)


def _get_loader(conf, train_set, val_set, test_set, rank=None):
    if "cuda" in conf['device']:
        num_workers = 0 if platform.system() == "Windows" else 8
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    if conf['env']['use_data_parallel']:
        train_sampler = DistributedSampler(train_set,
                                           num_replicas=conf['env']['world_size'],
                                           rank=rank,
                                           shuffle=True)
        valid_sampler = DistributedSampler(val_set,
                                           num_replicas=conf['env']['world_size'],
                                           rank=rank,
                                           shuffle=False)
        if test_set is not None:
            test_sampler = DistributedSampler(test_set,
                                              num_replicas=conf['env']['world_size'],
                                              rank=rank,
                                              shuffle=False)
        else:
            test_sampler = None

        train_loader_shuffle, val_loader_shuffle, test_loader_shuffle = False, False, False

    else:
        train_sampler, valid_sampler, test_sampler = None, None, None
        train_loader_shuffle, val_loader_shuffle, test_loader_shuffle = True, False, False

    train_loader = DataLoader(
        train_set,
        batch_size=conf['train']['batch_size'],
        shuffle=train_loader_shuffle,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=conf['train']['batch_size'],
        shuffle=val_loader_shuffle,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=valid_sampler,
    )
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=conf['train']['batch_size'],
            shuffle=test_loader_shuffle,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=test_sampler,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


def _get_torch_speech_commands(conf, device):
    train_set = SubsetSC(conf, "training")
    val_set = SubsetSC(conf, "validation")
    test_set = SubsetSC(conf, "testing")

    return _get_loader(conf, train_set, val_set, test_set, rank=device)


def _get_dataset(conf, device, with_waveform):
    if conf['data'].get('paths').get(conf['data']['type']) is not None \
            and conf['data'].get('paths').get(conf['data']['type']).get('h5') is not None \
            and not conf['data']['augmentation']['use_augmentation']:
        train_set = AudioDatasetH5(conf, mode='train', with_waveform=with_waveform)
        val_set = AudioDatasetH5(conf, mode='val', with_waveform=with_waveform)
        test_set = AudioDatasetH5(conf, mode='test', with_waveform=with_waveform)
    else:
        if conf['data']['augmentation']['use_augmentation']:
            logger.info('Using slow dataset (single files instead of h5), because data augmentation not supported for h5 files yet')
        else:
            logger.warning("Using slow dataset (single files instead of h5)")
        train_set = AudioDataset(conf, mode='train', augmentation=get_augmentation(conf))
        val_set = AudioDataset(conf, mode='val', augmentation=None)
        test_set = AudioDataset(conf, mode='test', augmentation=None)

    # optional, e.g. not needed for pretraining set
    if len(test_set) == 0:
        test_set = None

    if conf['data']['use_subset']:
        indices_train = torch.randperm(len(train_set)).tolist()
        indices_valid = torch.randperm(len(val_set)).tolist()
        indices_test = torch.randperm(len(test_set)).tolist()

        train_set = torch.utils.data.Subset(train_set, indices_train[:1024])
        val_set = torch.utils.data.Subset(val_set, indices_valid[:512])
        test_set = torch.utils.data.Subset(test_set, indices_test[:512])

    return _get_loader(conf, train_set, val_set, test_set, rank=device)


def get_loaders(conf, device, with_waveform=False):
    if conf['data']['dataset'] == 'torch-speech-commands':
        return _get_torch_speech_commands(conf, device)
    elif conf['data']['dataset'] == 'timit' or conf['data']['dataset'] == 'vox2' or conf['data'][
        'dataset'] == 'libri-speech':
        return _get_dataset(conf, device, with_waveform=with_waveform)
    else:
        raise AttributeError("Unknown dataset: {}".format(conf['data']['dataset']))
