import h5pickle
import torch
from pathlib import Path
from torch.utils.data import Dataset
from datasets.preprocessing import get_mfcc_preprocess_fn
import pandas as pd


class AudioDatasetH5(Dataset):

    def __init__(self, conf, mode, df_base_path='/workspace/data_pa/'):

        df_base_path = Path(df_base_path)
        meta_base_path = Path('datasets/dfs')

        if mode == 'train':
            h5_fp = df_base_path / conf['data']['paths']['h5']['train']
            md_fp = meta_base_path / conf['data']['paths']['h5']['metadata']['train']
        elif mode == 'val':
            h5_fp = df_base_path / conf['data']['paths']['h5']['val']
            md_fp = meta_base_path / conf['data']['paths']['h5']['metadata']['val']
        elif mode == 'test':
            h5_fp = df_base_path / conf['data']['paths']['h5']['test']
            md_fp = meta_base_path / conf['data']['paths']['h5']['metadata']['test']
        else:
            raise AttributeError("Unknown mode: {}".format(mode))

        self.metadata_df = pd.read_csv(md_fp)
        self.h5_file = h5pickle.File(str(h5_fp.resolve()), 'r', skip_cache=False)  # , libver='latest', swmr=True)
        self.preprocess = get_mfcc_preprocess_fn(mask_pos=conf['masking']['position'],
                                                 n_frames=conf['masking']['n_frames'],
                                                 k_frames=conf['masking']['k_frames'],
                                                 start_idx=conf['masking']['start_idx'])

        self.k_frames = conf['masking']['k_frames']
        self.n_frames = conf['masking']['n_frames']
        self.sliding_window = conf['masking']['start_idx'] == 'sliding-window'

        # ignore all files < k_frames + n_frames
        self.metadata_df = self.metadata_df[self.metadata_df['MFCC_length'] >= (self.n_frames + self.k_frames)]
        print("{} set has {} valid entries".format(mode, len(self.metadata_df)))

        # calculate new index mapping if with sliding window
        if self.sliding_window:
            self.sliding_window_indexes = {}  # mapping item -> [index_dataframe, mfcc_start, mfcc_end]
            item_count = 0
            for index, row in self.metadata_df.iterrows():
                length, start, end = row['MFCC_length'], row['MFCC_start'], row['MFCC_end']

                while self.n_frames + self.k_frames <= length:
                    assert start + self.n_frames + self.k_frames <= end
                    self.sliding_window_indexes[item_count] = [index, start, start + self.n_frames + self.k_frames]
                    start += self.n_frames + self.k_frames
                    length -= self.n_frames + self.k_frames
                    item_count += 1

        # calculate the dataset length
        if self.sliding_window:
            self.dataset_length = len(self.sliding_window_indexes)
        else:
            self.dataset_length = len(self.metadata_df)

    def __getitem__(self, item):

        if self.sliding_window:
            index_dataframe, mfcc_start, mfcc_end = self.sliding_window_indexes[item]
        else:
            index_dataframe = item
            mfcc_start = self.metadata_df['MFCC_start'][index_dataframe]
            mfcc_end = self.metadata_df['MFCC_end'][index_dataframe]

        speaker = self.metadata_df['Speaker'][index_dataframe]
        mfcc = self.h5_file['MFCC'][:, :, mfcc_start:mfcc_end]
        # mfcc_start, mfcc_end = self.h5_file['META'][item]
        # speaker = self.h5_file['Speaker'][item][0].decode('ascii')
        # filepath = self.h5_file['Filepath'][item][0].decode('ascii')

        mfcc = torch.from_numpy(mfcc)
        data, target = self.preprocess(mfcc)

        return data, target, mfcc, None, speaker

    def __len__(self):
        return self.dataset_length

    def __del__(self):
        self.h5_file.close()
