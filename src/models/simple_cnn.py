import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.encoder = self.get_encoder()
        self.center = self.get_center_block()
        self.decoder = self.decoder()


    def get_encoder(self):
        encoder = nn.Sequential(
            self.get_encoder_blocks([1, 32, 64, 64], (5, 5)),
            self.get_encoder_blocks([64, 64, 128, 128], (7, 7)),
            self.get_encoder_blocks([128, 128, 256, 256], (11, 11)),
        )
        return encoder


    def get_encoder_blocks(self, channels, kernel_size):
        blocks = nn.Sequential(
            self.get_encoder_block(channels[0], channels[1], kernel_size),
            self.get_encoder_block(channels[1], channels[2], kernel_size),
            self.get_encoder_block(channels[2], channels[3], kernel_size),
        )
        return blocks

    def get_encoder_block(self, in_channels, out_channels, kernel_size):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=(2, 2), bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        return block


    def get_center_block(self):
        return nn.Sequential(
            nn.Conv2d(256, 256, (1, 1)),
            nn.ReLU(inplace=True),
        )

    def get_decoder(self):
        encoder = nn.Sequential(
            self.get_decoder_blocks([256, 256, 128, 128], (11, 11)),
            self.get_decoder_blocks([128, 128, 64, 64], (11, 11)),
            self.get_decoder_blocks([64, 64, 32, 1], (11, 11)),
        )
        return encoder


    def get_decoder_blocks(self, channels, kernel_size):
        blocks = nn.Sequential(
            self.get_decoder_block(channels[0], channels[1], kernel_size),
            self.get_decoder_block(channels[1], channels[2], kernel_size),
            self.get_decoder_block(channels[2], channels[3], kernel_size),
        )
        return blocks

    def get_decoder_block(self, in_channels, out_channels, kernel_size):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=(2, 2), bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, x, y):
        x = self.encoder(x)
        x = self.center(x)
        print(x.shape)
        return self.decoder(x)
