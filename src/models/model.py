import logging

import torch
import torch.nn as nn

from .encoder.vgg import VGGExtractor

# mit TIMIT: https://github.com/hirofumi0810/neural_sp
# mit TIMIT: https://github.com/okkteam/Transformer-Transducer
# https://github.com/sooftware/Speech-Transformer
# https://github.com/kaituoxu/Speech-Transformer
# https://github.com/gentaiscool/end2end-asr-pytorch

logger = logging.getLogger(__name__)


def get_model(conf, device):
    # custom_encoder = VGGExtractor().to(conf['device'])
    # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    model = torch.nn.Transformer(d_model=40, nhead=8, num_encoder_layers=6,
                                 num_decoder_layers=6)  # , custom_encoder=custom_encoder) TODO: encoder not working!!!

    # print(model)
    print("Model has {} parameters".format(count_parameters(model)))

    if 'load_weights' in conf and conf['load_weights'] != 'None':
        load_weights(conf, model, device)

    return model.to(device)


def load_weights(net_config, model, rank=None):
    name = "{}.pth".format(net_config['load_weights'])
    mapping = torch.device(net_config['device']) if not isinstance(rank, int) else {'cuda:%d' % 0: 'cuda:%d' % rank}

    state_dict = torch.load("/workspace/data_pa/trained_models/{}".format(name), map_location=mapping)

    state_dict_single = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            state_dict_single[k[7:]] = v
        else:
            state_dict_single[k] = v

    model.load_state_dict(state_dict)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
