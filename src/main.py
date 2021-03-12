import platform

import torchaudio
import logging
from train import train
from evaluate import evaluate
from utils.conf_reader import get_config
from utils.log import setup_logging
import torch.multiprocessing as mp


# TODO: Read https://github.com/stdm/stdm.github.io/blob/master/downloads/papers/CISP_2009.pdf
# https://github.com/stdm/stdm.github.io/blob/master/downloads/papers/PhdThesis_2010.pdf

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
    logger.info(conf)

    if n_frames is not None:
        conf['n_frames'] = n_frames

    if conf['mode'] == "train":

        if conf['env']['use_data_parallel']:
            world_size = conf['env']['world_size']
            mp.spawn(train,
                     args=(world_size, conf),
                     nprocs=world_size,
                     join=True)
        else:
            train(None, None, conf)

    elif conf['mode'] == "eval":
        evaluate(conf)

    else:
        raise AttributeError("Unknown mode in config file: {}".format(conf.get('mode')))


if __name__ == '__main__':
    init()
    main()
    # from datasets.helper import create_h5_file_vox2
    # create_h5_file_vox2()
