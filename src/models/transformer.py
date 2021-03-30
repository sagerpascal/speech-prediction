import math
import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """
    source: https://towardsdatascience.com/stock-predictions-with-state-of-the-art-transformer-and-time-embeddings-3a4485237de6
    and:    https://arxiv.org/pdf/1907.05321.pdf
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim

        self.W = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.B = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.w = nn.Parameter(torch.Tensor(1, 1))
        self.b = nn.Parameter(torch.Tensor(input_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.W, 0, 1)
        nn.init.uniform_(self.B, 0, 1)
        nn.init.uniform_(self.w, 0, 1)
        nn.init.uniform_(self.b, 0, 1)

    def forward(self, x):
        x = torch.mean(x, dim=2, keepdim=True)
        original = self.w * x + self.b
        x = torch.repeat_interleave(x, self.output_dim, dim=-1)
        sin_trans = torch.sin(torch.matmul(x, self.W) + self.B)
        return torch.cat([sin_trans, original], -1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CustomTransformer(nn.Transformer):

    def __init__(self, conf, device, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(CustomTransformer, self).__init__(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                                num_decoder_layers=num_decoder_layers)
        self.conf = conf
        self.device = device
        self.k_frames = self.conf['masking']['k_frames']
        self.start_mask = 99999.
        self.x_pos_enc = Time2Vec(self.conf['masking']['n_frames'], d_model - 1)
        self.t_pos_enc = Time2Vec(self.conf['masking']['k_frames'], d_model - 1)
        self.x_pos_enc = PositionalEncoding(d_model)
        self.t_pos_enc = PositionalEncoding(d_model)

        self.out = nn.Linear(d_model, self.conf['data']['transform']['n_mfcc'])

    def forward(self, x, y):
        target = torch.ones_like(y) * self.start_mask
        target[:, 0, :] = x[:, -1, :]
        target[:, 1:, :] = y[:, :-1, :]

        x = self.x_pos_enc(x)
        target = self.t_pos_enc(target)
        x2 = x.permute(1, 0, 2)
        target2 = target.permute(1, 0, 2)
        target_mask = super().generate_square_subsequent_mask(target2.shape[0]).to(self.device)
        result = super().forward(x2, target2, tgt_mask=target_mask)
        return self.out(result.permute(1, 0, 2))

        # result = torch.ones((x.shape[0], self.k_frames + 1, x.shape[2])).to(self.device) * self.start_mask
        # result[:, 0, :] = x[:, -1, :]
        # for i in range(self.k_frames):
        #     target = result[:, :i + 1, :]
        #     x = self.x_pos_enc(x)
        #     target = self.t_pos_enc(target)
        #     x2 = x.permute(1, 0, 2)
        #     target2 = target.permute(1, 0, 2)
        #     pred = super().forward(x2, target2)
        #     pred = pred.permute(1, 0, 2)
        #     result[:, i + 1, :] = pred[:, -1, :]
        #
        # result = result[:, 1:, :]
        # return self.out(result)

    def predict(self, x):
        result = torch.ones((x.shape[0], self.k_frames+1, x.shape[2])).to(self.device) * self.start_mask
        result[:, 0, :] = x[:, -1, :]
        for i in range(self.k_frames):
            with torch.no_grad():
                target = result[:, :i + 1, :]
                x = self.x_pos_enc(x)
                target = self.t_pos_enc(target)
                pred = super().forward(x.permute(1, 0, 2), target.permute(1, 0, 2))
                pred = pred.permute(1, 0, 2)
                result[:, i+1, :] = pred[:, -1, :]

        return self.out(result[:, 1:, :])