import torch
import torchaudio

from datasets.torch_speech_commands import SubsetSC, collate_fn, label_to_index


def _get_torch_speech_commands(conf):
    """ For debugging purposes """

    train_set = SubsetSC("training")
    val_set = SubsetSC("validation")
    test_set = SubsetSC("testing")

    batch_size = 64

    train_labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    val_labels = sorted(list(set(datapoint[2] for datapoint in val_set)))
    test_labels = sorted(list(set(datapoint[2] for datapoint in test_set)))
    train_collate_fn = collate_fn(train_labels, label_to_index)
    val_collate_fn = collate_fn(val_labels, label_to_index)
    test_collate_fn = collate_fn(test_labels, label_to_index)


    if "cuda" in conf['device']:
        num_workers = 0 # TODO: set to >=1 if not Windows
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=val_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=test_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    assert len(train_labels) == len(val_labels)
    assert len(val_labels) == len(test_labels)

    n_input = 1
    n_output = len(train_labels)

    return train_loader, val_loader, test_loader, n_input, n_output



def get_loaders(conf):
    if conf['dataset'] == 'torch-speech-commands':
        return _get_torch_speech_commands(conf)
    else:
        raise AttributeError("Unknown dataset: {}".format(conf['dataset']))


