import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import SpeechTransformer
from .transformer.sublayers import VGGExtractor

# mit TIMIT: https://github.com/hirofumi0810/neural_sp
# mit TIMIT: https://github.com/okkteam/Transformer-Transducer
# https://github.com/sooftware/Speech-Transformer
# https://github.com/kaituoxu/Speech-Transformer
# https://github.com/gentaiscool/end2end-asr-pytorch

logger = logging.getLogger(__name__)


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


def get_model(conf, n_input, n_output):
    if 'load_model' in conf and conf['load_model'] != 'None':
        try:
            model = torch.load("data/trained_models/{}.pth".format(conf['load_model']),
                               map_location=torch.device(conf['device']))
        except FileNotFoundError:
            model = torch.load("../data/trained_models/{}.pth".format(conf['load_model']),
                               map_location=torch.device(conf['device']))

    else:
        # model = M5(n_input=n_input, n_output=n_output)
        # model = SpeechTransformer(num_classes=35, d_model=512, num_heads=8, input_dim=40, extractor='vgg')

        # use a smaller model...
        # TODO: nume_classes=35 for classification
        # model = SpeechTransformer(num_classes=40*40, d_model=1600, num_heads=2, input_dim=40, extractor='vgg', num_encoder_layers=3, num_decoder_layers=3)

        custom_encoder = VGGExtractor()
        model = torch.nn.Transformer(d_model=512, custom_encoder=custom_encoder) # d_model=40 if without encoder

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
