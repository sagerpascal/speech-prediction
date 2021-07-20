import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Prenet(nn.Module):
    """ The Pre-Net of the GRU model """

    def __init__(self, input_size, conf):
        super(Prenet, self).__init__()
        num_layers = conf['num_layers']
        hidden_size = conf['hidden_size']
        dropout = conf['dropout']

        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList(
            [self.get_block(in_features=in_size, out_features=out_size, dropout=dropout)
             for (in_size, out_size) in zip(input_sizes, output_sizes)])

    def get_block(self, in_features, out_features, dropout):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_features),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class Postnet(nn.Module):
    """ The Post-Net of the GRU model """

    def __init__(self, conf, input_size, output_size, out_features_postnet):
        super(Postnet, self).__init__()

        postnet_conf = conf['model']['gru']['prenet']
        num_layers = postnet_conf['num_layers']
        hidden_size = postnet_conf['hidden_size']
        dropout = postnet_conf['dropout']
        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList(
            [self.get_block(in_features=in_size, out_features=out_size, dropout=dropout)
             for (in_size, out_size) in zip(input_sizes, output_sizes)])

        final_bias = conf['data']['stats'][conf['data']['type']]['train']['mean']
        self.apply_resize = conf['masking']['start_idx'] != 'full'
        self.layer_seq_len = nn.Linear(in_features=conf['masking']['n_frames'], out_features=out_features_postnet)
        self.activation = nn.ReLU()
        self.layer_dim = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=1, stride=1,
                                   bias=final_bias)

    def get_block(self, in_features, out_features, dropout):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_features),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = torch.transpose(x, 1, 2)
        if self.apply_resize:
            x = self.activation(self.layer_seq_len(x))

        outputs = torch.transpose(self.layer_dim(x), 1, 2)
        return outputs


class GRUModel(nn.Module):
    """ The Main Model """

    def __init__(self, conf):
        super(GRUModel, self).__init__()

        self.conf = conf
        self.prenet_conf = conf['model']['gru']['prenet']
        self.rnn_conf = conf['model']['gru']['rnn']
        self.out_features_postnet = int(conf['masking']['k_frames'] / conf['model']['gru']['refeed_fac'])
        self.k_frames = conf['masking']['k_frames']
        self.rnn_hidden_dim = self.rnn_conf['hidden_size']

        feature_dim = conf['data']['transform']['n_mfcc'] if conf['data']['type'] == 'mfcc' else \
            conf['data']['transform']['n_mels']
        feature_dim_in = feature_dim + 2 if conf['masking']['add_metadata'] else feature_dim
        self.feature_dim_out = feature_dim

        if self.prenet_conf['use_prenet']:
            rnn_input_size = self.prenet_conf['hidden_size']
            assert rnn_input_size == self.rnn_conf['hidden_size']
            self.prenet = Prenet(
                input_size=feature_dim_in,
                conf=self.prenet_conf)
        else:
            rnn_input_size = feature_dim_in
            self.prenet = None

        in_sizes = ([rnn_input_size] +
                    [self.rnn_conf['hidden_size']] * (self.rnn_conf['num_layers'] - 1))
        out_sizes = [self.rnn_conf['hidden_size']] * self.rnn_conf['num_layers']
        self.rnns = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True, bidirectional=True)
             for (in_size, out_size) in zip(in_sizes, out_sizes)])

        self.rnn_dropout = nn.Dropout(self.rnn_conf['dropout'])
        self.rnn_residual = self.rnn_conf['use_residual']

        self.postnet = Postnet(
            conf,
            input_size=self.rnn_hidden_dim,
            output_size=self.feature_dim_out,
            out_features_postnet=self.out_features_postnet)

    def forward(self, x, target=None, seq_lengths=None, epoch=None):

        inputs = x.clone()
        predicted_mel = None

        seq_len = inputs.size(1)

        # currently, only fix lengths are used
        if seq_lengths is None or seq_lengths[0] is None:
            seq_lengths = torch.IntTensor(inputs.shape[0] * [self.conf['masking']['n_frames']])

        for cycle in range(self.conf['model']['gru']['refeed_fac']):

            if self.prenet is not None:
                rnn_inputs = self.prenet(inputs)
            else:
                rnn_inputs = inputs

            rnn_inputs = pack_padded_sequence(rnn_inputs, seq_lengths, True)
            hidden = torch.autograd.Variable(torch.zeros(2, inputs.shape[0], self.rnn_hidden_dim)).cuda()

            for i, layer in enumerate(self.rnns):
                rnn_outputs, hidden = layer(rnn_inputs, hidden)

                rnn_outputs, _ = pad_packed_sequence(rnn_outputs, True, total_length=seq_len)
                rnn_outputs = (rnn_outputs[:, :, :self.rnn_hidden_dim] + rnn_outputs[:, :, self.rnn_hidden_dim:])

                if i + 1 < len(self.rnns):
                    # apply dropout except the last rnn layer
                    rnn_outputs = self.rnn_dropout(rnn_outputs)

                rnn_inputs, _ = pad_packed_sequence(rnn_inputs, True, total_length=seq_len)

                if self.rnn_residual and rnn_inputs.size(-1) == rnn_outputs.size(-1):
                    rnn_outputs = rnn_outputs + rnn_inputs

                rnn_inputs = pack_padded_sequence(rnn_outputs, seq_lengths, True)

            pr = self.postnet(rnn_outputs)

            if self.conf['masking']['start_idx'] == 'full':
                predicted_mel = pr
            else:
                if predicted_mel is None:
                    predicted_mel = torch.zeros((pr.shape[0], self.k_frames, self.feature_dim_out),
                                                dtype=torch.float32).to('cuda')
                predicted_mel[:, self.out_features_postnet * cycle:self.out_features_postnet * (cycle + 1),
                :] = pr.clone()

            inputs = inputs.clone()
            if self.conf['masking']['add_metadata']:
                inputs[:, :-self.out_features_postnet, :] = inputs[:, self.out_features_postnet:, :].clone()
                inputs[:, -self.out_features_postnet:, :-2] = predicted_mel[:,
                                                              self.out_features_postnet * cycle:self.out_features_postnet * (
                                                                      cycle + 1), :].clone()
            else:
                inputs[:, :-self.out_features_postnet, :] = inputs[:, self.out_features_postnet:, :].clone()
                inputs[:, -self.out_features_postnet:, :] = predicted_mel[:,
                                                            self.out_features_postnet * cycle:self.out_features_postnet * (
                                                                    cycle + 1), :].clone()

        return predicted_mel

    def predict(self, x, y=None, seq_lengths=None):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x, seq_lengths=seq_lengths)

        return x
