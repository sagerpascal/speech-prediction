import os
from pathlib import Path

import h5pickle
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from datasets.normalization import zero_norm, undo_zero_norm
from datasets.preprocessing import get_frames_preprocess_fn
from utils.conf_reader import get_config


class AudioDatasetH5(Dataset):

    def __init__(self, conf, mode, h5_base_path='/workspace/data_pa/', with_waveform=False):

        h5_base_path = Path(h5_base_path)
        meta_base_path = Path('datasets/dfs')

        if mode == 'train':
            ds_fp = meta_base_path / conf['data']['paths']['df']['train']
            if conf['data']['type'] == 'mfcc':
                h5_fp = h5_base_path / conf['data']['paths']['mfcc']['h5']['train']
                md_fp = meta_base_path / conf['data']['paths']['mfcc']['h5']['metadata']['train']
            elif conf['data']['type'] == 'mel-spectro':
                h5_fp = h5_base_path / conf['data']['paths']['mel-spectro']['h5']['train']
                md_fp = meta_base_path / conf['data']['paths']['mel-spectro']['h5']['metadata']['train']
        elif mode == 'val':
            ds_fp = meta_base_path / conf['data']['paths']['df']['val']
            if conf['data']['type'] == 'mfcc':
                h5_fp = h5_base_path / conf['data']['paths']['mfcc']['h5']['val']
                md_fp = meta_base_path / conf['data']['paths']['mfcc']['h5']['metadata']['val']
            elif conf['data']['type'] == 'mel-spectro':
                h5_fp = h5_base_path / conf['data']['paths']['mel-spectro']['h5']['val']
                md_fp = meta_base_path / conf['data']['paths']['mel-spectro']['h5']['metadata']['val']
        elif mode == 'test':
            ds_fp = meta_base_path / conf['data']['paths']['df']['test']
            if conf['data']['type'] == 'mfcc':
                h5_fp = h5_base_path / conf['data']['paths']['mfcc']['h5']['test']
                md_fp = meta_base_path / conf['data']['paths']['mfcc']['h5']['metadata']['test']
            elif conf['data']['type'] == 'mel-spectro':
                h5_fp = h5_base_path / conf['data']['paths']['mel-spectro']['h5']['test']
                md_fp = meta_base_path / conf['data']['paths']['mel-spectro']['h5']['metadata']['test']
        else:
            raise AttributeError("Unknown mode: {}".format(mode))

        self.metadata_df = pd.read_csv(md_fp)

        # std and mean from training set
        if conf['data']['type'] == 'mfcc':
            self.mean = conf['data']['stats']['mfcc']['train']['mean']
            self.std = conf['data']['stats']['mfcc']['train']['std']
            self.length_key, self.start_key, self.end_key = 'MFCC_length', 'MFCC_start', 'MFCC_end'
            self.file_key = 'MFCC'

        elif conf['data']['type'] == 'mel-spectro':
            self.mean = conf['data']['stats']['mel-spectro']['train']['mean']
            self.std = conf['data']['stats']['mel-spectro']['train']['std']
            self.length_key, self.start_key = 'mel_spectro_length', 'mel_spectro_start'
            self.end_key, self.file_key = 'mel_spectro_end', 'Mel-Spectrogram'

        self.preprocess = get_frames_preprocess_fn(mask_pos=conf['masking']['position'],
                                                   n_frames=conf['masking']['n_frames'],
                                                   k_frames=conf['masking']['k_frames'],
                                                   start_idx=conf['masking']['start_idx'])

        self.h5_file = h5pickle.File(str(h5_fp.resolve()), 'r', skip_cache=False)  # , libver='latest', swmr=True)

        self.k_frames = conf['masking']['k_frames']
        self.n_frames = conf['masking']['n_frames']
        self.window_shift = conf['masking']['window_shift']
        self.sliding_window = conf['masking']['start_idx'] == 'sliding-window'
        self.with_waveform = with_waveform

        # ignore all files < k_frames + n_frames
        valid_idx = self.metadata_df[self.length_key] >= (self.n_frames + self.k_frames)
        self.metadata_df = self.metadata_df[valid_idx]

        if self.with_waveform:
            self.dataset_df = pd.read_csv(ds_fp)
            self.dataset_df = self.dataset_df[valid_idx]

        print("{} set has {} valid entries".format(mode, len(self.metadata_df)))

        # calculate new index mapping if with sliding window
        if self.sliding_window:
            self.sliding_window_indexes = {}  # mapping item -> [index_dataframe, start_key, end_key]
            item_count = 0
            for index, row in self.metadata_df.iterrows():
                length, start, end = row[self.length_key], row[self.start_key], row[self.end_key]

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
            index_dataframe, start_idx, end_idx = self.sliding_window_indexes[item]
        else:
            index_dataframe = item
            start_idx = self.metadata_df[self.start_key][index_dataframe]
            end_idx = self.metadata_df[self.end_key][index_dataframe]

        if self.with_waveform:
            file = self.dataset_df['file_path'][index_dataframe]
            waveform = torchaudio.load(file)[0]
        else:
            waveform = None

        speaker = self.metadata_df['Speaker'][index_dataframe]
        complete_data = self.h5_file[self.file_key][:, :, start_idx:end_idx]  # mfcc or mel-spectro
        complete_data = zero_norm(complete_data, self.mean, self.std)

        complete_data = torch.from_numpy(complete_data)
        data, target = self.preprocess(complete_data)

        return data, target, complete_data, waveform, speaker

    def __len__(self):
        return self.dataset_length

    # def __del__(self):
    #     self.h5_file.close()


def play_some_data():
    import random
    import scipy.io.wavfile
    from librosa.feature.inverse import mfcc_to_audio

    conf = get_config()
    dataset = AudioDatasetH5(conf, 'train', with_waveform=False)
    mean, std = conf['data']['stats']['mfcc']['train']['mean'], conf['data']['stats']['mfcc']['train']['std']

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


def test_sliding_window():
    conf = get_config()
    conf['masking']['window_shift'] = 60
    conf['masking']['n_frames'] = 30
    conf['masking']['k_frames'] = 20
    mean = conf['data']['stats'][conf['data']['type']]['train']['mean']
    std = conf['data']['stats'][conf['data']['type']]['train']['std']

    h5_fp = Path('/workspace/data_pa/') / conf['data']['paths']['h5']['train']
    h5_file = h5pickle.File(str(h5_fp.resolve()), 'r', skip_cache=False)
    mfcc_orig = h5_file['MFCC'][:, :, 0:170]

    dataset = AudioDatasetH5(conf, 'train', with_waveform=False)

    data_0, target_0, mfcc_0, *_ = dataset[0]
    data_1, target_1, mfcc_1, *_ = dataset[1]

    data_0, target_0, mfcc_0 = data_0.cpu().numpy(), target_0.cpu().numpy(), mfcc_0.cpu().numpy()
    data_1, target_1, mfcc_1 = data_1.cpu().numpy(), target_1.cpu().numpy(), mfcc_1.cpu().numpy()

    data_0 = undo_zero_norm(data_0, mean, std)
    target_0 = undo_zero_norm(target_0, mean, std)
    mfcc_0 = undo_zero_norm(mfcc_0, mean, std)
    data_1 = undo_zero_norm(data_1, mean, std)
    target_1 = undo_zero_norm(target_1, mean, std)
    mfcc_1 = undo_zero_norm(mfcc_1, mean, std)

    assert np.allclose(mfcc_orig[:, :, :conf['masking']['n_frames']], data_0, atol=0.0001)
    assert np.allclose(
        mfcc_orig[:, :, conf['masking']['n_frames']:conf['masking']['n_frames'] + conf['masking']['k_frames']],
        target_0, atol=0.0001)
    assert np.allclose(mfcc_orig[:, :, :conf['masking']['n_frames'] + conf['masking']['k_frames']], mfcc_0, atol=0.0001)

    assert np.allclose(
        mfcc_orig[:, :, conf['masking']['window_shift']:conf['masking']['window_shift'] + conf['masking']['n_frames']],
        data_1, atol=0.0001)
    assert np.allclose(mfcc_orig[:, :,
                       conf['masking']['window_shift'] + conf['masking']['n_frames']:conf['masking']['window_shift'] +
                                                                                     conf['masking']['n_frames'] +
                                                                                     conf['masking']['k_frames']],
                       target_1, atol=0.0001)
    assert np.allclose(mfcc_orig[:, :,
                       conf['masking']['window_shift']:conf['masking']['window_shift'] + conf['masking']['n_frames'] +
                                                       conf['masking']['k_frames']], mfcc_1, atol=0.0001)


if __name__ == '__main__':
    os.chdir('../')
    test_sliding_window()
    play_some_data()
