import platform

import torchaudio

from train import train
from evaluate import evaluate
from utils.conf_reader import get_config
from utils.log import setup_logging, setup_wandb


def init():
    setup_logging()
    if platform.system() == "Windows":
        torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
        torchaudio.set_audio_backend("soundfile")
    else:
        torchaudio.set_audio_backend("sox_io")


def main():
    conf = get_config()
    if conf['use_wandb']:
        setup_wandb()

    if conf['mode'] == "train":
        train(conf)
    elif conf['mode'] == "eval":
        evaluate(conf)
    else:
        raise AttributeError("Unknown mode in config file: {}".format(conf.get('mode')))


if __name__ == '__main__':
    init()
    main()
