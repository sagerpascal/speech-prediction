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


def get_model(conf):
    if 'load_model' in conf and conf['load_model'] != 'None':
        try:
            model = torch.load("data/trained_models/{}.pth".format(conf['load_model']),
                               map_location=torch.device(conf['device']))
        except FileNotFoundError:
            model = torch.load("../data/trained_models/{}.pth".format(conf['load_model']),
                               map_location=torch.device(conf['device']))

    else:
        # custom_encoder = VGGExtractor().to(conf['device'])
        # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        model = torch.nn.Transformer(d_model=40) #, custom_encoder=custom_encoder) TODO: encoder not working!!!

        print(model)
        print("Model has {} parameters".format(count_parameters(model)))

    if 'cpu' in conf['device'] and isinstance(model, torch.nn.DataParallel):
        model = model.module

    elif 'cuda' in conf['device'] and torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        model = torch.nn.DataParallel(model)

    elif 'cuda' in conf['device'] and isinstance(model, torch.nn.DataParallel) and len(
            model.device_ids) != torch.cuda.device_count():
        model.device_ids = range(torch.cuda.device_count())

    return model.to(conf['device'])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
