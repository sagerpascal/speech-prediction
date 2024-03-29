import argparse
from pathlib import Path

import torch
import yaml


def _read_conf_file(name):
    """ Read the config.yaml file """

    base_paths = [Path('configs'), Path('../configs'), Path('src') / 'configs']

    for p in base_paths:
        try:
            file = yaml.load(open(p / name), Loader=yaml.FullLoader)
            return file
        except FileNotFoundError:
            pass

    raise FileNotFoundError("Config file not found")


def get_config():
    """ Parse the arguments and merge them with the config file """

    conf_file = _read_conf_file('config.yaml')
    conf_file_ds = _read_conf_file(conf_file['data']['config_file'])

    conf_file['data'] = {**conf_file['data'], **conf_file_ds}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='train', help="'train' or 'eval'")
    parser.add_argument("--lr", default=0.0001, help="The learning rate")
    parser.add_argument("--weight_decay", default=0.0001, help="Weight decay of the optimizer")
    parser.add_argument("--load_weights", default=None, help="name of the model to load")
    parser.add_argument("--batch_size", default=32, help="The mini-batch size")
    parser.add_argument("--step_size", default=20, help="LR scheduler step size")
    parser.add_argument("--gamma", default=0.8, help="LR scheduler gamma")
    parser.add_argument("--gru_num_prenet_layer", default=5, help="Number of prenet layers")
    parser.add_argument("--gru_num_rnn_layer", default=4, help="Number of RNN layers")
    parser.add_argument("--gru_num_postnet_layer", default=3, help="Number of postnet layers")
    args = parser.parse_args()

    args_dict = {
        'load_weights': str(args.load_weights),
        'device': device,
        'mode': str(args.mode),
    }

    conf = {**conf_file, **args_dict}
    conf['optimizer']['lr'] = float(args.lr)
    conf['optimizer']['weight_decay'] = float(args.weight_decay)
    conf['lr_scheduler']['step_size'] = int(args.step_size)
    conf['lr_scheduler']['gamma'] = float(args.gamma)
    conf['train']['batch_size'] = int(args.batch_size)
    conf['env']['use_data_parallel'] = 'cuda' in device and conf_file['env']['world_size'] > 1
    conf['masking']['window_shift'] = conf['masking']['n_frames'] + conf['masking']['k_frames'] if \
        conf['masking']['window_shift'] == 'None' else conf['masking']['window_shift']
    conf['model']['gru']['prenet']['num_layers'] = int(args.gru_num_prenet_layer)
    conf['model']['gru']['rnn']['num_layers'] = int(args.gru_num_rnn_layer)

    if conf['model']['type'] == 'gru':
        assert conf['model']['gru']['refeed_fac'] >= 1
        assert conf['masking']['k_frames'] % conf['model']['gru']['refeed_fac'] == 0

    if conf['masking']['start_idx'] == 'full':
        assert conf['model']['gru']['refeed_fac'] == 1

    return conf
