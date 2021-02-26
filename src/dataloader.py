import platform

from torch.utils.data import DataLoader

from datasets.torch_speech_commands import SubsetSC, collate_fn


def _get_torch_speech_commands(conf):
    """ For debugging purposes """

    train_set = SubsetSC("training")
    val_set = SubsetSC("validation")
    test_set = SubsetSC("testing")

    if "cuda" in conf['device']:
        num_workers = 0 if platform.system() == "Windows" else 2
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    train_loader = DataLoader(
        train_set,
        batch_size=conf['batch_size'],
        shuffle=True,
        collate_fn=collate_fn(),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=conf['batch_size'],
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn(),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=conf['batch_size'],
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn(),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_loaders(conf):
    if conf['dataset'] == 'torch-speech-commands':
        return _get_torch_speech_commands(conf)
    else:
        raise AttributeError("Unknown dataset: {}".format(conf['dataset']))



if __name__ == '__main__':
    import torchaudio
    import sounddevice as sd
    from librosa.feature.inverse import mfcc_to_audio
    train_set = SubsetSC("training")
    data = train_set[1]
    waveform = data[0]
    mfcc_transf = torchaudio.transforms.MFCC(melkwargs={"hop_length": 512})
    mfcc = mfcc_transf(waveform).numpy().squeeze()
    sd.play(waveform.T, 16000, blocking=True)
    sd.play(mfcc_to_audio(mfcc, norm='ortho').T, 16000, blocking=True)



