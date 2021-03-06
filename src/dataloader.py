import platform

from torch.utils.data import DataLoader
from datasets.torch_speech_commands import SubsetSC
from datasets.dataset import AudioDataset
from datasets.collate import collate_fn

def _get_loader(conf, train_set, val_set, test_set):

    if "cuda" in conf['device']:
        num_workers = 0 if platform.system() == "Windows" else 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    train_loader = DataLoader(
        train_set,
        batch_size=conf['train']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn(conf),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=conf['train']['batch_size'],
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn(conf),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=conf['train']['batch_size'],
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn(conf),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader

def _get_torch_speech_commands(conf):

    train_set = SubsetSC("training")
    val_set = SubsetSC("validation")
    test_set = SubsetSC("testing")

    return _get_loader(conf, train_set, val_set, test_set)

def _get_dataset(conf):
    train_set = AudioDataset(conf, mode='train')
    val_set = AudioDataset(conf, mode='val')
    test_set = AudioDataset(conf, mode='test')

    return _get_loader(conf, train_set, val_set, test_set)

def get_loaders(conf):
    if conf['data']['dataset'] == 'torch-speech-commands':
        return _get_torch_speech_commands(conf)
    elif conf['data']['dataset'] == 'timit':
        return _get_dataset(conf)
    else:
        raise AttributeError("Unknown dataset: {}".format(conf['data']['dataset']))




