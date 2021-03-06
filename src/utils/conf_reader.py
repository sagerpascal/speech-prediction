import yaml
import argparse
import torch
from pathlib import Path

def _read_conf_file(name):
    base_paths = [Path('configs'), Path('../configs'),  Path('src') / 'configs']

    for p in base_paths:
        try:
            file = yaml.load(open(p / name), Loader=yaml.FullLoader)
            return file
        except FileNotFoundError:
            pass

    raise FileNotFoundError("Config file not found")


def get_config():
    conf_file = _read_conf_file('config.yaml')
    conf_file_ds = _read_conf_file(conf_file['data']['config_file'])

    conf_file['data'] = {**conf_file['data'], **conf_file_ds}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ddp = 'cuda' in device and conf_file['world_size'] > 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", default=0.005, help="The learning rate")
    parser.add_argument("--load_model", default=None, help="path to the model to load")
    args = parser.parse_args()

    args_dict = {
        'load_model': str(args.load_model),
        'device': device,
        'use_data_parallel': ddp,
    }

    conf = {**conf_file, **args_dict}
    conf['optimizer']['lr'] = float(args.learning_rate)

    return conf