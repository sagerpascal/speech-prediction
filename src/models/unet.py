import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class CustomUnet(smp.UnetPlusPlus):
    def __init__(self, conf):
        super().__init__(
            encoder_name='efficientnet-b2',  # 'vgg11',
            # encoder_depth=5,  # 5
            # decoder_channels=[128, 256, 256, 512, 512], # reomve
            encoder_weights=None,
            in_channels=1,
            classes=1,
        )

        assert np.floor(np.log(conf['masking']['n_frames']) / np.log(2)) >= 5  # otherwise encoder is too deep

        self.conf = conf
        self.conv2d = nn.Conv2d(self.decoder.out_channels[-1], 1, kernel_size=3, padding=3 // 2)
        self.out_mapping = nn.Linear(conf['masking']['n_frames'], conf['masking']['k_frames'], bias=True)

    def forward(self, x, y=None):
        features = self.encoder(x.unsqueeze(1))
        decoder_output = self.decoder(*features)
        decoder_output = self.conv2d(decoder_output)
        return self.out_mapping(decoder_output.squeeze(1).permute(0, 2, 1)).permute(0, 2, 1)





