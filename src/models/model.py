import logging

import torch

from models.transformer import CustomTransformer
from models.unet import CustomUnet
from models.simple_cnn import SimpleCNN
from models.apc import APCModel

# mit TIMIT: https://github.com/hirofumi0810/neural_sp
# mit TIMIT: https://github.com/okkteam/Transformer-Transducer
# https://github.com/sooftware/Speech-Transformer
# https://github.com/kaituoxu/Speech-Transformer
# https://github.com/gentaiscool/end2end-asr-pytorch

logger = logging.getLogger(__name__)


def get_model(conf, device):
    print("{} GPU's available".format(torch.cuda.device_count()))

    if conf['model']['type'] == 'transformer':
        d_model = conf['data']['transform']['n_mfcc'] if conf['data']['type'] == 'mfcc' else conf['data']['transform'][
            'n_mels']
        model = CustomTransformer(conf, device,
                                  d_model=d_model,
                                  nhead=conf['model']['transformer']['n_heads'],
                                  num_encoder_layers=conf['model']['transformer']['n_encoder_layers'],
                                  num_decoder_layers=conf['model']['transformer']['n_decoder_layers'])

    elif conf['model']['type'] == 'unet':
        model = CustomUnet(conf)
    elif conf['model']['type'] == 'cnn':
        model = SimpleCNN(conf)
    elif conf['model']['type'] == 'apc':
        model = APCModel(conf)
    else:
        raise AttributeError("Unknown Model in config file: {}".format(conf['model']['type']))

    print(model)
    # print(summary(model.to(device), (1, 60, 128), 32))
    print("Model has {} parameters".format(count_parameters(model)))

    if 'load_weights' in conf and conf['load_weights'] != 'None':
        load_weights(conf, model, device)

    return model.to(device)


def load_weights(net_config, model, rank=None):
    name = "{}.pth".format(net_config['load_weights'])
    mapping = torch.device(net_config['device']) if not isinstance(rank, int) else {'cuda:%d' % 0: 'cuda:%d' % rank}

    try:
        state_dict = torch.load("/workspace/data_pa/trained_models/{}".format(name), map_location=mapping)
    except:
        state_dict = torch.load("trained_models/{}".format(name), map_location=mapping)

    state_dict_single = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            state_dict_single[k[7:]] = v
        else:
            state_dict_single[k] = v

    model.load_state_dict(state_dict_single, strict=False) # TODO: remove strict=False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
