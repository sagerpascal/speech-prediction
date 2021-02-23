from utils.conf_reader import get_config
from utils.log import setup_logging, setup_wandb
from train import train
import torchaudio

def init():
    setup_logging()
    torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
    torchaudio.set_audio_backend("soundfile")

def main():
    conf = get_config()
    if conf['use_wandb']:
        setup_wandb()

    if conf['mode'] == "train":
        train(conf)
    else:
        raise AttributeError("Unknown mode in config file: {}".format(conf.get('mode')))



if __name__ == '__main__':
    init()
    main()
