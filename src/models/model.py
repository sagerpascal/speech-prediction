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
            try:
                model = torch.load("../data/trained_models/{}.pth".format(conf['load_model']),
                               map_location=torch.device(conf['device']))
            except FileNotFoundError:
                model = torch.load("trained_models/{}.pth".format(conf['load_model']),
                                   map_location=torch.device(conf['device']))

    else:
        # custom_encoder = VGGExtractor().to(conf['device'])
        # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        model = torch.nn.Transformer(d_model=40) #, custom_encoder=custom_encoder) TODO: encoder not working!!!

        print(model)
        print("Model has {} parameters".format(count_parameters(model)))

    #  just for "old" models using DataParallel (now using DistributedDataParallel)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
