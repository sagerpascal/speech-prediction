import os
from pathlib import Path

import h5pickle
import numpy as np
import pandas as pd
import torch
import torchaudio
import logging
import platform
from torch.utils.data import Dataset
from audio_datasets.preprocessing import get_mfcc_transform, get_mel_spectro_transform
from audio_datasets.normalization import zero_norm, undo_zero_norm
from audio_datasets.preprocessing import get_frames_preprocess_fn
from utils.conf_reader import get_config

logger = logging.getLogger(__name__)

class AudioDatasetH5(Dataset):

    def __init__(self, conf, mode, h5_base_path='/workspace/data_pa/', with_waveform=False):

        if platform.system() == "Windows":
            h5_base_path = 'D:/Projekte/temporal-speech-context/data/'

        h5_base_path = Path(h5_base_path)
        meta_base_path = Path('audio_datasets/dfs')
        ds_fp = meta_base_path / conf['data']['paths']['df'][mode]
        h5_fp = h5_base_path / conf['data']['paths'][conf['data']['type']]['h5'][mode]
        md_fp = meta_base_path / conf['data']['paths'][conf['data']['type']]['h5']['metadata'][mode]

        if not md_fp.exists():
            logger.warning("Dataset for mode {} not defined".format(mode))
            self.dataset_length = 0
        else:
            speakers_df = pd.read_csv(meta_base_path / conf['data']['paths'].get('speakers'))
            sentences_df = pd.read_csv(meta_base_path / conf['data']['paths'].get('sentences'))
            self.speaker_to_id = pd.Series(speakers_df.id.values, index=speakers_df.speaker).to_dict()
            self.id_to_speaker = pd.Series(speakers_df.speaker.values, index=speakers_df.id).to_dict()
            self.sentence_to_id = pd.Series(sentences_df.id.values, index=sentences_df.sentence).to_dict()
            self.id_to_sentence = pd.Series(sentences_df.sentence.values, index=sentences_df.id).to_dict()
            self.use_metadata = conf['masking']['add_metadata']

            self.metadata_df = pd.read_csv(md_fp)

            # std and mean from training set
            if conf['data']['type'] == 'raw':
                self.length_key, self.start_key, self.end_key = 'raw_length', 'raw_start', 'raw_end'
                self.file_key = 'RAW'
                self.use_norm = False
                self.shape_len = 2

            elif conf['data']['type'] == 'mfcc':
                self.length_key, self.start_key, self.end_key = 'MFCC_length', 'MFCC_start', 'MFCC_end'
                self.file_key = 'MFCC'
                self.use_norm = True
                self.shape_len = 3

            elif conf['data']['type'] == 'mel-spectro':
                self.length_key, self.start_key = 'mel_spectro_length', 'mel_spectro_start'
                self.end_key, self.file_key = 'mel_spectro_end', 'Mel-Spectrogram'
                self.use_norm = True
                self.shape_len = 3

            self.mean = conf['data'].get('stats').get(conf['data']['type']).get('train').get('mean')
            self.std = conf['data'].get('stats').get(conf['data']['type']).get('train').get('std')

            if self.use_norm and (self.mean is None or self.std is None):
                logger.warning("Cannot use global normalization: Mean and/or Std not defined")
                self.use_norm = False

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

                    while self.n_frames + self.k_frames <= length:
                        assert start + self.n_frames + self.k_frames <= end
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

        speaker = self.metadata_df['speaker'][index_dataframe]
        sentence = self.metadata_df['sentence'][index_dataframe]

        start_seq, end_seq = self.h5_file['META'][index_dataframe]
        assert start_idx >= start_seq and start_idx <= end_seq
        assert end_idx >= start_seq and end_idx <= end_seq

        if self.shape_len == 2:
            complete_data = self.h5_file[self.file_key][:, start_idx:end_idx]  # raw
        elif self.shape_len == 3:
            complete_data = self.h5_file[self.file_key][:, :, start_idx:end_idx]  # mfcc or mel-spectro

        complete_data = torch.from_numpy(complete_data)

        if self.use_norm:
            complete_data = zero_norm(complete_data, self.mean, self.std)

        if self.shape_len == 2:
            data, target = self.preprocess(complete_data.unsqueeze(0))
            data = data.squeeze(0)
            target = target.squeeze(0)
        else:
            data, target = self.preprocess(complete_data)

        if self.use_metadata:
            data_ = torch.ones(data.shape[0], data.shape[1]+2, data.shape[2], dtype=torch.float32)
            data_[:, :-2, :] = data
            data_[:, -2, :] *= self.speaker_to_id[speaker]
            data_[:, -1, :] *= self.sentence_to_id[sentence]
            return data_, target, complete_data, waveform, speaker
        else:
            return data, target, complete_data, waveform, speaker

    def __len__(self):
        return self.dataset_length

    # def __del__(self):
    #     self.h5_file.close()


def play_some_data(type='mfcc'):
    import random
    import scipy.io.wavfile
    from librosa.feature.inverse import mfcc_to_audio, mel_to_audio

    conf = get_config()
    dataset = AudioDatasetH5(conf, 'train', with_waveform=True)

    if type == 'mfcc':
        mean, std = conf['data']['stats']['mfcc']['train']['mean'], conf['data']['stats']['mfcc']['train']['std']
        to_audio = mfcc_to_audio
    elif type == 'mel-spectro':
        mean, std = conf['data']['stats']['mfcc']['train']['mean'], conf['data']['stats']['mfcc']['train']['std']
        to_audio = mel_to_audio

    for i in range(3):
        idx = random.randint(0, len(dataset))

        for j in range(20):
            # two following segments
            data, target, mfcc, waveform, speaker, start, stop, path, idx2 = dataset[idx + j]

            data = data.squeeze(dim=0).cpu().numpy()
            target = target.squeeze(dim=0).cpu().numpy()
            mfcc = mfcc.squeeze(dim=0).cpu().numpy()

            if type == 'mfcc' or type == 'mel-spectro':
                mfcc = undo_zero_norm(mfcc, mean, std)
                data = undo_zero_norm(data, mean, std)
                target = undo_zero_norm(target, mean, std)
                mfcc_audio = to_audio(mfcc, hop_length=conf['data']['transform']['hop_length'])
                data_audio = to_audio(data, hop_length=conf['data']['transform']['hop_length'])
                target_audio = to_audio(target, hop_length=conf['data']['transform']['hop_length'])
            else:
                mfcc_audio = mfcc.T
                data_audio = data.T
                target_audio = target.T

            print("{}".format(path))
            scipy.io.wavfile.write('testfiles_dataset/waveform_{}-{}.wav'.format(i, j),
                                   conf['data']['transform']['sample_rate'], waveform.cpu().numpy().T)
            scipy.io.wavfile.write('testfiles_dataset/MFCC_{}-{}.wav'.format(i, j),
                                   conf['data']['transform']['sample_rate'],
                                   mfcc_audio)
            scipy.io.wavfile.write('testfiles_dataset/data_{}-{}.wav'.format(i, j),
                                   conf['data']['transform']['sample_rate'],
                                   data_audio)
            scipy.io.wavfile.write('testfiles_dataset/target_{}-{}.wav'.format(i, j),
                                   conf['data']['transform']['sample_rate'],
                                   target_audio)


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
    # test_sliding_window()
    play_some_data(type='raw')
