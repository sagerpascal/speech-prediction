import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Prenet(nn.Module):
    """Prenet is a multi-layer fully-connected network with ReLU activations.
    During training and testing (i.e., feature extraction), each input frame is
    passed into the Prenet, and the Prenet output is then fed to the RNN. If
    Prenet configuration is None, the input frames will be directly fed to the
    RNN without any transformation.
    """

    def __init__(self, input_size, num_layers, hidden_size, dropout):
        super(Prenet, self).__init__()
        input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        output_sizes = [hidden_size] * num_layers

        self.layers = nn.ModuleList(
            [nn.Linear(in_features=in_size, out_features=out_size)
             for (in_size, out_size) in zip(input_sizes, output_sizes)])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, mel_dim)
        for layer in self.layers:
            inputs = self.dropout(self.relu(layer(inputs)))

        return inputs
        # inputs: (batch_size, seq_len, out_dim)


class Postnet(nn.Module):
    """Postnet is a simple linear layer for predicting the target frames given
    the RNN context during training. We don't need the Postnet for feature
    extraction.
    """

    def __init__(self, conf, input_size, output_size):
        super(Postnet, self).__init__()
        # TODO: cleanup
        self.layer_seq_len = nn.Linear(in_features=conf['masking']['n_frames'], out_features=conf['masking']['k_frames'])
        self.layer_dim = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=1, stride=1)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, hidden_size)
        inputs = torch.transpose(inputs, 1, 2)
        inputs = self.layer_seq_len(inputs)
        # inputs: (batch_size, hidden_size, seq_len) -- for conv1d operation

        outputs = torch.transpose(self.layer_dim(inputs), 1, 2)
        return outputs
        # (batch_size, seq_len, output_size) -- back to the original shape


class APCModel(nn.Module):
    """This class defines Autoregressive Predictive Coding (APC), a model that
    learns to extract general speech features from unlabeled speech data. These
    features are shown to contain rich speaker and phone information, and are
    useful for a wide range of downstream tasks such as speaker verification
    and phone classification.
    An APC model consists of a Prenet (optional), a multi-layer GRU network,
    and a Postnet. For each time step during training, the Prenet transforms
    the input frame into a latent representation, which is then consumed by
    the GRU network for generating internal representations across the layers.
    Finally, the Postnet takes the output of the last GRU layer and attempts to
    predict the target frame.
    After training, to extract features from the data of your interest, which
    do not have to be i.i.d. with the training data, simply feed-forward the
    the data through the APC model, and take the the internal representations
    (i.e., the GRU hidden states) as the extracted features and use them in
    your tasks.
    """

    def __init__(self, conf):
        super(APCModel, self).__init__()

        self.conf = conf
        self.prenet_conf = conf['model']['apc']['prenet']
        self.rnn_conf = conf['model']['apc']['rnn']
        feature_dim = conf['data']['transform']['n_mfcc'] if conf['data']['type'] == 'mfcc' else \
            conf['data']['transform']['n_mels']

        if self.prenet_conf['use_prenet']:
            rnn_input_size = self.prenet_conf['hidden_size']
            assert rnn_input_size == self.rnn_conf['hidden_size']
            self.prenet = Prenet(
                input_size=feature_dim,
                num_layers=self.prenet_conf['num_layers'],
                hidden_size=self.prenet_conf['hidden_size'],
                dropout=self.prenet_conf['dropout'])
        else:
            rnn_input_size = feature_dim
            self.prenet = None

        in_sizes = ([rnn_input_size] +
                    [self.rnn_conf['hidden_size']] * (self.rnn_conf['num_layers'] - 1))
        out_sizes = [self.rnn_conf['hidden_size']] * self.rnn_conf['num_layers']
        self.rnns = nn.ModuleList(
            [nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True)
             for (in_size, out_size) in zip(in_sizes, out_sizes)])

        self.rnn_dropout = nn.Dropout(self.rnn_conf['dropout'])
        self.rnn_residual = self.rnn_conf['use_residual']

        self.postnet = Postnet(
            conf,
            input_size=self.rnn_conf['hidden_size'],
            output_size=feature_dim)

    def forward(self, inputs, target=None):
        """Forward function for both training and testing (feature extraction).
        input:
          inputs: (batch_size, seq_len, mel_dim)
        return:
          predicted_mel: (batch_size, seq_len, mel_dim)
          internal_reps: (num_layers + x, batch_size, seq_len, rnn_hidden_size),
            where x is 1 if there's a prenet, otherwise 0
        """
        seq_len = inputs.size(1)

        # currently, only fix lengths are used
        seq_lengths = torch.IntTensor(inputs.shape[0] * [self.conf['masking']['n_frames']])

        if self.prenet is not None:
            rnn_inputs = self.prenet(inputs)
            # rnn_inputs: (batch_size, seq_len, rnn_input_size)
            internal_reps = [rnn_inputs]
            # also include prenet_outputs in internal_reps
        else:
            rnn_inputs = inputs
            internal_reps = []

        # TODO: necessary???
        packed_rnn_inputs = pack_padded_sequence(rnn_inputs, seq_lengths, True)

        for i, layer in enumerate(self.rnns):
            packed_rnn_outputs, _ = layer(packed_rnn_inputs)

            # TODO: necessary???
            rnn_outputs, _ = pad_packed_sequence(
                packed_rnn_outputs, True, total_length=seq_len)
            # outputs: (batch_size, seq_len, rnn_hidden_size)

            if i + 1 < len(self.rnns):
                # apply dropout except the last rnn layer
                rnn_outputs = self.rnn_dropout(rnn_outputs)

            # TODO: necessary???
            rnn_inputs, _ = pad_packed_sequence(
                packed_rnn_inputs, True, total_length=seq_len)
            # rnn_inputs: (batch_size, seq_len, rnn_hidden_size)

            if self.rnn_residual and rnn_inputs.size(-1) == rnn_outputs.size(-1):
                # Residual connections
                rnn_outputs = rnn_outputs + rnn_inputs

            internal_reps.append(rnn_outputs)

            # TODO: necessary???
            packed_rnn_inputs = pack_padded_sequence(rnn_outputs, seq_lengths, True)

        predicted_mel = self.postnet(rnn_outputs)
        # predicted_mel: (batch_size, seq_len, mel_dim)

        internal_reps = None# TODO internal_reps = torch.stack(internal_reps)
        return predicted_mel, internal_reps
        # predicted_mel is only for training; internal_reps is the extracted
        # features

    def predict(self, x, y=None):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x
