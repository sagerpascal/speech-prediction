import pandas as pd
from pathlib import Path
import torchaudio
from torch.utils.data import Dataset
from datasets.preprocessing import get_mfcc_transform, get_mfcc_preprocess_fn
from datasets.normalization import zero_norm


# http://www.openslr.org/12/
# https://lionbridge.ai/datasets/best-speech-recognition-datasets-for-machine-learning/
# https://www.sciencedirect.com/science/article/pii/S0885230819302712


class AudioDataset(Dataset):

    def __init__(self, conf, mode, df_base_path='datasets/dfs'):

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

        self.mean, self.std = conf['data']['stats']['train']['mean'], conf['data']['stats']['train']['std']

        self.mfcc_transform = get_mfcc_transform(conf).to('cuda')
        self.preprocess = get_mfcc_preprocess_fn(mask_pos=conf['masking']['position'],
                                                 n_frames=conf['masking']['n_frames'],
                                                 k_frames=conf['masking']['k_frames'],
                                                 start_idx=conf['masking']['start_idx'])

        self.k_frames = conf['masking']['k_frames']
        self.n_frames = conf['masking']['n_frames']
        self.window_shift = conf['masking']['window_shift']
        self.sliding_window = conf['masking']['start_idx'] == 'sliding-window'

        # ignore all files < k_frames + n_frames
        self.df = self.df[self.df['MFCC_length'] >= (self.n_frames + self.k_frames)]
        print("{} set has {} valid entries".format(mode, len(self.df)))

        # calculate new index mapping if with sliding window
        if self.sliding_window:
            self.sliding_window_indexes = {}  # mapping item -> [index_dataframe, mfcc_start, mfcc_end]
            item_count = 0
            for index, row in self.df.iterrows():
                length, start = row['MFCC_length'], 0

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
            index_dataframe, mfcc_start, mfcc_end = self.sliding_window_indexes[item]
        else:
            index_dataframe = item

        waveform = torchaudio.load(self.df['file_path'][index_dataframe])
        speaker = self.df['speaker'][index_dataframe]

        mfcc = self.mfcc_transform(waveform[0])
        if self.sliding_window:
            mfcc = mfcc[:, :, mfcc_start:mfcc_end]

        mfcc = zero_norm(mfcc, self.mean, self.std)  # normalize
        data, target = self.preprocess(mfcc)

        return data, target, mfcc, waveform[0], speaker

    def __len__(self):
        return self.dataset_length
