import platform
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from datasets.torch_speech_commands import SubsetSC
from datasets.dataset import AudioDataset
from datasets.collate import collate_fn


def _get_loader(conf, train_set, val_set, test_set, rank=None):
    if "cuda" in conf['device']:
        num_workers = 0 if platform.system() == "Windows" else 4
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
        test_sampler = DistributedSampler(test_set,
                                          num_replicas=conf['env']['world_size'],
                                          rank=rank,
                                          shuffle=False)

        train_loader_shuffle, val_loader_shuffle, test_loader_shuffle = False, False, False

    else:
        train_sampler, valid_sampler, test_sampler = None, None, None
        train_loader_shuffle, val_loader_shuffle, test_loader_shuffle = True, False, False

    train_loader = DataLoader(
        train_set,
        batch_size=conf['train']['batch_size'],
        shuffle=train_loader_shuffle,
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
    test_loader = DataLoader(
        test_set,
        batch_size=conf['train']['batch_size'],
        shuffle=test_loader_shuffle,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=test_sampler,
    )

    return train_loader, val_loader, test_loader


def _get_torch_speech_commands(conf, device):
    train_set = SubsetSC(conf, "training")
    val_set = SubsetSC(conf, "validation")
    test_set = SubsetSC(conf, "testing")

    return _get_loader(conf, train_set, val_set, test_set, rank=device)


def _get_dataset(conf, device):
    train_set = AudioDataset(conf, mode='train')
    val_set = AudioDataset(conf, mode='val')
    test_set = AudioDataset(conf, mode='test')

    return _get_loader(conf, train_set, val_set, test_set, rank=device)


def get_loaders(conf, device):
    if conf['data']['dataset'] == 'torch-speech-commands':
        return _get_torch_speech_commands(conf, device)
    elif conf['data']['dataset'] == 'timit' or conf['data']['dataset'] == 'vox2':
        return _get_dataset(conf, device)
    else:
        raise AttributeError("Unknown dataset: {}".format(conf['data']['dataset']))
