import pandas as pd
from pathlib import Path
import torchaudio
import logging
from torch.utils.data import Dataset
from audio_datasets.preprocessing import get_mfcc_transform, get_mel_spectro_transform, get_frames_preprocess_fn
from audio_datasets.normalization import zero_norm
import torch
import h5pickle

logger = logging.getLogger(__name__)


class DbMelSpectroTransform:

    def __init__(self, conf):
        self.conf = conf
        self.mel_spectro_transform = get_mel_spectro_transform(conf).to('cpu')
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, data):
        return self.to_db(self.mel_spectro_transform(data))


class AudioDataset(Dataset):

    def __init__(self, conf, mode, df_base_path='audio_datasets/dfs', augmentation=None):
        self.conf = conf
        self.augmentation = augmentation
        df_base_path = Path(df_base_path)
        df_fp = df_base_path / conf['data']['paths']['df'][mode]

        if not df_fp.exists():
            logger.warning("Dataset for mode {} not defined".format(mode))
            self.dataset_length = 0

        else:
            speakers_df = pd.read_csv(df_base_path / conf['data']['paths'].get('speakers'))
            sentences_df = pd.read_csv(df_base_path / conf['data']['paths'].get('sentences'))
            self.speaker_to_id = pd.Series(speakers_df.id.values, index=speakers_df.speaker).to_dict()
            self.id_to_speaker = pd.Series(speakers_df.speaker.values, index=speakers_df.id).to_dict()
            self.sentence_to_id = pd.Series(sentences_df.id.values, index=sentences_df.sentence).to_dict()
            self.id_to_sentence = pd.Series(sentences_df.sentence.values, index=sentences_df.id).to_dict()
            self.use_metadata = conf['masking']['add_metadata']

            try:
                # use h5 file instead of multiple audio files if available for better performance
                h5_raw_fp = Path('/workspace/data_pa/') / conf['data']['paths'].get('raw').get('h5').get(mode)
                self.h5_raw_file = h5pickle.File(str(h5_raw_fp.resolve()), 'r', skip_cache=False)
                h5_raw_md_fp = Path('audio_datasets/dfs') / conf['data']['paths']['raw']['h5']['metadata'][mode]
                self.h5_raw_metadata_df = pd.read_csv(h5_raw_md_fp)
                logger.info("Using h5-file instead of single audio files for better performance")
            except Exception as e:
                print(e)
                self.h5_raw_file = None
                self.h5_raw_metadata_df = None

            self.df = pd.read_csv(df_fp)

            if conf['data']['type'] == 'raw':
                self.transform = None
                self.length_key = 'raw_length'
                self.use_norm = False
                self.shape_len = 2

            elif conf['data']['type'] == 'mfcc':
                self.transform = get_mfcc_transform(conf).to('cuda')
                self.length_key = 'MFCC_length'
                self.use_norm = True
                self.shape_len = 3

            elif conf['data']['type'] == 'mel-spectro':
                self.transform = DbMelSpectroTransform(conf)
                self.length_key = 'mel_spectro_length'
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

            self.k_frames = conf['masking']['k_frames']
            self.n_frames = conf['masking']['n_frames']
            self.window_shift = conf['masking']['window_shift']
            self.sliding_window = conf['masking']['start_idx'] == 'sliding-window'

            # ignore all files < k_frames + n_frames
            self.df = self.df[self.df[self.length_key] >= (self.n_frames + self.k_frames)]
            logger.info("{} set has {} valid entries".format(mode, len(self.df)))

            # calculate new index mapping if with sliding window
            if self.sliding_window:
                self.sliding_window_indexes = {}  # mapping item -> [index_dataframe, mfcc_start, mfcc_end]
                item_count = 0
                for index, row in self.df.iterrows():
                    length, start = row[self.length_key], 0

                    while self.n_frames + self.k_frames <= length:
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

        if self.h5_raw_file is not None and self.h5_raw_metadata_df is not None:
            # start_idx_h5 = self.h5_raw_metadata_df['raw_start'][index_dataframe]
            # end_idx_h5 = self.h5_raw_metadata_df['raw_end'][index_dataframe]
            # TODO: this is only valid for new raw h5-files (distinguish between old and new files) and then copy this to dataset_h5
            length_h5 = self.h5_raw_metadata_df['raw_length'][index_dataframe]
            waveform = self.h5_raw_file['RAW'][index_dataframe, :length_h5]
            waveform = torch.as_tensor(waveform, dtype=torch.float32)
            complete_data = waveform.unsqueeze(0)
        else:
            waveform = torchaudio.load(self.df['file_path'][index_dataframe])
            complete_data = waveform[0]

        speaker = self.df['speaker'][index_dataframe]
        sentence = self.df['sentence'][index_dataframe]

        if self.augmentation is not None:
            complete_data = self.augmentation(complete_data)

        if self.transform is not None:
            complete_data = self.transform(complete_data)  # either mfcc or mel-spectrogram

        if self.sliding_window:
            if self.shape_len == 2:
                complete_data = complete_data[:, start_idx:end_idx]
            elif self.shape_len == 3:
                complete_data = complete_data[:, :, start_idx:end_idx]

        if self.use_norm:
            complete_data = zero_norm(complete_data, self.mean, self.std)  # normalize

        if self.shape_len == 2:
            data, target = self.preprocess(complete_data.unsqueeze(0))
            data = data.squeeze(0)
            target = target.squeeze(0)
        else:
            data, target = self.preprocess(complete_data)

        if self.use_metadata:
            data_ = torch.ones(data.shape[0], data.shape[1] + 2, data.shape[2], dtype=torch.float32)
            data_[:, :-2, :] = data
            data_[:, -2, :] *= self.speaker_to_id[speaker]
            data_[:, -1, :] *= self.sentence_to_id[sentence]
            return data_, target, complete_data, waveform, speaker
        else:
            return data, target, complete_data, waveform, speaker

    def __len__(self):
        return self.dataset_length
