import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from optimizer import get_optimizer, get_lr
from dataloader import get_loaders
from utils.conf_reader import get_config
from models.model import count_parameters
from audio_datasets.preprocessing import get_mfcc_transform, get_mel_spectro_transform
from losses.loss import get_loss
from utils.meter import AverageValueMeter

class ModelHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(768, 512, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Linear(85, 256),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 81)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layers(x)
        return x


class ModelPipeline(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.head = ModelHead()
        self.head.to('cuda')
        print(count_parameters(self.head))

    def forward(self, x):
        features = self.feature_extractor([x[i, :].numpy() for i in range(4)], return_tensors='pt', sampling_rate=16000)
        embeddings = self.model(features['input_values'])
        pred = self.head(embeddings['last_hidden_state'].to('cuda'))
        return pred


def main():
    conf = get_config()
    device = 'cuda'
    loader_train, loader_val, _ = get_loaders(conf, device)
    model = ModelPipeline()
    optimizer = get_optimizer(conf, model)
    loss_func = get_loss(conf)
    loss_meter = AverageValueMeter()

    with tqdm(loader_train, file=sys.stdout) as iterator:
        for x, y in iterator:
            x, y = x, y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_func(y_pred, y)

            loss.backward()
            optimizer.step()
            loss_meter.add(loss.cpu().detach().numpy())

            iterator.set_postfix_str("loss = {}".format(loss_meter.mean))



if __name__ == '__main__':
    os.chdir('../')
    main()
