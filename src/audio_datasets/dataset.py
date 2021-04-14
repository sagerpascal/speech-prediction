import pandas as pd
from pathlib import Path
import torchaudio
from torch.utils.data import Dataset
from audio_datasets.preprocessing import get_mfcc_transform, get_mel_spectro_transform, get_frames_preprocess_fn
from audio_datasets.normalization import zero_norm


# http://www.openslr.org/12/
# https://lionbridge.ai/datasets/best-speech-recognition-datasets-for-machine-learning/
# https://www.sciencedirect.com/science/article/pii/S0885230819302712


class AudioDataset(Dataset):

    def __init__(self, conf, mode, df_base_path='audio_datasets/dfs'):

        self.conf = conf
        df_base_path = Path(df_base_path)

        if mode == 'train':
            df_fp = df_base_path / conf['data']['paths']['df']['train']
        elif mode == 'val':
            df_fp = df_base_path / conf['data']['paths']['df']['val']
        elif mode == 'test':
            df_fp = df_base_path / conf['data']['paths']['df']['test']
        else:
            raise AttributeError("Unknown mode: {}".format(mode))

        self.df = pd.read_csv(df_fp)

        if conf['data']['type'] == 'mfcc':
            self.mean = conf['data']['stats']['mfcc']['train']['mean']
            self.std = conf['data']['stats']['mfcc']['train']['std']
            self.transform = get_mfcc_transform(conf).to('cuda')
            self.length_key = 'MFCC_length'

        elif conf['data']['type'] == 'mel-spectro':
            self.mean = conf['data']['stats']['mel-spectro']['train']['mean']
            self.std = conf['data']['stats']['mel-spectro']['train']['std']
            self.transform = get_mel_spectro_transform(conf).to('cuda')

        self.preprocess = get_frames_preprocess_fn(mask_pos=conf['masking']['position'],
                                                   n_frames=conf['masking']['n_frames'],
                                                   k_frames=conf['masking']['k_frames'],
                                                   start_idx=conf['masking']['start_idx'])

        self.k_frames = conf['masking']['k_frames']
        self.n_frames = conf['masking']['n_frames']
        self.window_shift = conf['masking']['window_shift']
        self.sliding_window = conf['masking']['start_idx'] == 'sliding-window'

        # ignore all files < k_frames + n_frames
        self.df = self.df[self.df[self.length_key] >= (self.n_frames + self.k_frames)]
        print("{} set has {} valid entries".format(mode, len(self.df)))

        # calculate new index mapping if with sliding window
        if self.sliding_window:
            self.sliding_window_indexes = {}  # mapping item -> [index_dataframe, mfcc_start, mfcc_end]
            item_count = 0
            for index, row in self.df.iterrows():
                length, start = row[self.length_key], 0

                while self.window_shift <= length:
                    assert start + self.window_shift <= length
                    self.sliding_window_indexes[item_count] = [index, start, start + self.n_frames + self.k_frames]
                    start += self.window_shift
                    length -= self.window_shift
                    item_count += 1

        # calculate the dataset length
        if self.sliding_window:
            self.dataset_length = len(self.sliding_window_indexes)
        else:
            self.dataset_length = len(self.df)

    def __getitem__(self, item):

        if self.sliding_window:
            index_dataframe, start_idx, end_idx = self.sliding_window_indexes[item]
        else:
            index_dataframe = item

        waveform = torchaudio.load(self.df['file_path'][index_dataframe])
        speaker = self.df['speaker'][index_dataframe]

        complete_data = self.transform(waveform[0])  # either mfcc or mel-spectrogram
        if self.sliding_window:
            complete_data = complete_data[:, :, start_idx:end_idx]

        complete_data = zero_norm(complete_data, self.mean, self.std)  # normalize
        data, target = self.preprocess(complete_data)

        return data, target, complete_data, waveform[0], speaker

    def __len__(self):
        return self.dataset_length
