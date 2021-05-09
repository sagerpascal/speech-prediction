import logging
import platform
import random

import numpy as np
import torch
import torch.multiprocessing as mp
import torchaudio

from evaluate import evaluate
from train import train
from utils.conf_reader import get_config
from utils.log import setup_logging


# TODO: Read https://github.com/stdm/stdm.github.io/blob/master/downloads/papers/CISP_2009.pdf
# https://github.com/stdm/stdm.github.io/blob/master/downloads/papers/PhdThesis_2010.pdf

def init():
    setup_logging()
    if platform.system() == "Windows":
        torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
        torchaudio.set_audio_backend("soundfile")
    else:
        torchaudio.set_audio_backend("sox_io")

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


def main(n_frames=None, k_frames=None, window_shift=None):
    logger = logging.getLogger(__name__)
    conf = get_config()
    logger.info(conf)

    if n_frames is not None:
        conf['masking']['n_frames'] = n_frames

    if k_frames is not None:
        conf['masking']['k_frames'] = k_frames

    if window_shift is not None:
        conf['masking']['window_shift'] = window_shift

    if conf['mode'] == "train":

        if conf['env']['use_data_parallel']:
            world_size = conf['env']['world_size']
            mp.spawn(train,
                     args=(str(random.randint(12100, 65000)), random.randint(12100, 65000), world_size, conf),
                     nprocs=world_size,
                     join=True)
        else:
            train(None, None, None, conf)

    elif conf['mode'] == "eval":
        evaluate(conf)

    else:
        raise AttributeError("Unknown mode in config file: {}".format(conf.get('mode')))


if __name__ == '__main__':
    init()
    main()
