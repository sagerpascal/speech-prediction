import h5pickle
import torch
from pathlib import Path
from torch.utils.data import Dataset
from datasets.preprocessing import get_mfcc_preprocess_fn
import pandas as pd
import torchaudio
import os
import numpy as np
from datasets.normalization import zero_norm, undo_zero_norm


class AudioDatasetH5(Dataset):

    def __init__(self, conf, mode, h5_base_path='/workspace/data_pa/', with_waveform=False):

        h5_base_path = Path(h5_base_path)
        meta_base_path = Path('datasets/dfs')

        if mode == 'train':
            h5_fp = h5_base_path / conf['data']['paths']['h5']['train']
            ds_fp = meta_base_path / conf['data']['paths']['df']['train']
            md_fp = meta_base_path / conf['data']['paths']['h5']['metadata']['train']
        elif mode == 'val':
            h5_fp = h5_base_path / conf['data']['paths']['h5']['val']
            ds_fp = meta_base_path / conf['data']['paths']['df']['val']
            md_fp = meta_base_path / conf['data']['paths']['h5']['metadata']['val']
        elif mode == 'test':
            h5_fp = h5_base_path / conf['data']['paths']['h5']['test']
            ds_fp = meta_base_path / conf['data']['paths']['df']['test']
            md_fp = meta_base_path / conf['data']['paths']['h5']['metadata']['test']
        else:
            raise AttributeError("Unknown mode: {}".format(mode))

        self.metadata_df = pd.read_csv(md_fp)

        # std and mean from training set
        self.mean, self.std = conf['data']['stats']['train']['mean'], conf['data']['stats']['train']['std']

        self.h5_file = h5pickle.File(str(h5_fp.resolve()), 'r', skip_cache=False)  # , libver='latest', swmr=True)
        self.preprocess = get_mfcc_preprocess_fn(mask_pos=conf['masking']['position'],
                                                 n_frames=conf['masking']['n_frames'],
                                                 k_frames=conf['masking']['k_frames'],
                                                 start_idx=conf['masking']['start_idx'])

        self.k_frames = conf['masking']['k_frames']
        self.n_frames = conf['masking']['n_frames']
        self.window_shift = conf['masking']['window_shift']
        self.sliding_window = conf['masking']['start_idx'] == 'sliding-window'
        self.with_waveform = with_waveform

        # ignore all files < k_frames + n_frames
        valid_idx = self.metadata_df['MFCC_length'] >= (self.n_frames + self.k_frames)
        self.metadata_df = self.metadata_df[valid_idx]

        if self.with_waveform:
            self.dataset_df = pd.read_csv(ds_fp)
            self.dataset_df = self.dataset_df[valid_idx]

        print("{} set has {} valid entries".format(mode, len(self.metadata_df)))

        # calculate new index mapping if with sliding window
        if self.sliding_window:
            self.sliding_window_indexes = {}  # mapping item -> [index_dataframe, mfcc_start, mfcc_end]
            item_count = 0
            for index, row in self.metadata_df.iterrows():
                length, start, end = row['MFCC_length'], row['MFCC_start'], row['MFCC_end']

                while self.window_shift <= length:
                    assert start + self.window_shift <= end
                    self.sliding_window_indexes[item_count] = [index, start, start + self.n_frames + self.k_frames]
                    start += self.window_shift
                    length -= self.window_shift
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

        if self.with_waveform:
            file = self.dataset_df['file_path'][index_dataframe]
            waveform = torchaudio.load(file)[0]
        else:
            waveform = None

        speaker = self.metadata_df['Speaker'][index_dataframe]
        mfcc = self.h5_file['MFCC'][:, :, mfcc_start:mfcc_end]
        mfcc = zero_norm(mfcc, self.mean, self.std)

        # mfcc_start, mfcc_end = self.h5_file['META'][item]
        # speaker = self.h5_file['Speaker'][item][0].decode('ascii')
        # filepath = self.h5_file['Filepath'][item][0].decode('ascii')

        mfcc = torch.from_numpy(mfcc)
        data, target = self.preprocess(mfcc)

        return data, target, mfcc, waveform, speaker

    def __len__(self):
        return self.dataset_length

    # def __del__(self):
    #     self.h5_file.close()


def get_some_data():
    import random
    from utils.conf_reader import get_config
    import scipy.io.wavfile
    from librosa.feature.inverse import mfcc_to_audio

    conf = get_config()
    dataset = AudioDatasetH5(conf, 'train', with_waveform=True)
    mean, std = conf['data']['stats']['train']['mean'], conf['data']['stats']['train']['std']

    for i in range(3):
        idx = random.randint(0, len(dataset))

        for j in range(2):
            # two following segments
            data, target, mfcc, waveform, *_ = dataset[idx + j]

            data = data.squeeze(dim=0).cpu().numpy()
            target = target.squeeze(dim=0).cpu().numpy()
            mfcc = mfcc.squeeze(dim=0).cpu().numpy()

            mfcc = undo_zero_norm(mfcc, mean, std)
            data = undo_zero_norm(data, mean, std)
            target = undo_zero_norm(target, mean, std)

            print("{}".format(waveform))
            # scipy.io.wavfile.write('testfiles_dataset/waveform_{}-{}.wav'.format(i, j), conf['data']['transform']['sample_rate'], waveform.T)
            scipy.io.wavfile.write('testfiles_dataset/MFCC_{}-{}.wav'.format(i, j),
                                   conf['data']['transform']['sample_rate'],
                                   mfcc_to_audio(mfcc, hop_length=conf['data']['transform']['hop_length']))
            scipy.io.wavfile.write('testfiles_dataset/data_{}-{}.wav'.format(i, j),
                                   conf['data']['transform']['sample_rate'],
                                   mfcc_to_audio(data, hop_length=conf['data']['transform']['hop_length']))
            scipy.io.wavfile.write('testfiles_dataset/target_{}-{}.wav'.format(i, j),
                                   conf['data']['transform']['sample_rate'],
                                   mfcc_to_audio(target, hop_length=conf['data']['transform']['hop_length']))


if __name__ == '__main__':
    os.chdir('../')
    get_some_data()
