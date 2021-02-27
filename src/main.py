import platform

import torchaudio
import logging
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


def main(n_frames=None):
    logger = logging.getLogger(__name__)

    conf = get_config()

    if n_frames is not None:
        conf['n_frames'] = n_frames

    wandb_run = None
    if conf['use_wandb']:
        wandb_run = setup_wandb()

    if conf['mode'] == "train":
        train_logs, valid_logs = train(conf)
        logger.info('Training logs: {}\n Validation logs: {}'.format(str(train_logs), str(valid_logs)))

    elif conf['mode'] == "eval":
        evaluate(conf)

    if wandb_run is not None:
        wandb_run.finish()

    else:
        raise AttributeError("Unknown mode in config file: {}".format(conf.get('mode')))


if __name__ == '__main__':
    init()
    main()
