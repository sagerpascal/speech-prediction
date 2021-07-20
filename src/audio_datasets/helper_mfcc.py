import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from audio_datasets.preprocessing import get_mfcc_transform
from utils.conf_reader import get_config

logger = logging.getLogger(__name__)


def _create_df_metadata_vox2_h5(h5_fp, df_name):
    h5_file = h5py.File(h5_fp, 'r', libver='latest', swmr=True)

    print("Create metadata for file {}".format(h5_file))

    mfcc_starts, mfcc_ends, mfcc_lengths, speakers = [], [], [], []

    for i in tqdm(range(len(h5_file['META']))):
        mfcc_start, mfcc_end = h5_file['META'][i]
        speaker = h5_file['Speaker'][i][0].decode('ascii')
        mfcc_length = mfcc_end - mfcc_start

        if not mfcc_length == 0:
            mfcc_starts.append(mfcc_start)
            mfcc_ends.append(mfcc_end)
            mfcc_lengths.append(mfcc_length)
            speakers.append(speaker)

    df = pd.DataFrame.from_dict({
        'MFCC_start': mfcc_starts,
        'MFCC_end': mfcc_ends,
        'MFCC_length': mfcc_lengths,
        'Speaker': speakers
    })

    df.to_csv(r'{}.csv'.format(df_name), index=False)


def create_metadata_h5_vox2():
    base_path = Path('/workspace/data_pa/VOX2')
    _create_df_metadata_vox2_h5(base_path / 'vox2_mfcc_train.h5', 'vox2_metadata_train')
    _create_df_metadata_vox2_h5(base_path / 'vox2_mfcc_valid.h5', 'vox2_metadata_valid')
    _create_df_metadata_vox2_h5(base_path / 'vox2_mfcc_test.h5', 'vox2_metadata_test')


def add_to_file(i, mfcc_start, mfcc, speaker, filename, mfcc_d, speaker_d, filepath_d, meta_d, n_features, vox2=True,
                length_idx=2):
    mfcc_end = mfcc_start + mfcc.shape[length_idx]
    mfcc_d.resize((1, n_features, mfcc_end))
    mfcc_d[:, :, mfcc_start:mfcc_end] = mfcc
    speaker_d[i, :] = speaker.encode("ascii")
    filepath_d[i, :] = (speaker + "/" + filename).encode("ascii") if vox2 else filename.encode("ascii")
    meta_d[i, :] = np.array([mfcc_start, mfcc_end])
    return mfcc_end


def _create_h5_file_timit(conf, df_fp, h5_name, df_name):
    df = pd.read_csv(df_fp)
    mfcc_transform = get_mfcc_transform(conf).to('cuda')
    number_of_entries = len(df)

    f_h5 = h5py.File(h5_name, 'w', libver='latest')
    mfcc_dataset = f_h5.create_dataset('MFCC', (1, conf['data']['transform']['n_mfcc'], 1),
                                       maxshape=(1, conf['data']['transform']['n_mfcc'], None), chunks=True,
                                       dtype='float32')
    speaker_dataset = f_h5.create_dataset('Speaker', (number_of_entries, 1), 'S7')
    filepath_dataset = f_h5.create_dataset('Filepath', (number_of_entries, 1), 'S30')
    meta_dataset = f_h5.create_dataset('META', (number_of_entries, 2), dtype='int64')

    mfcc_start = 0
    indexes, mfcc_starts, mfcc_ends, mfcc_lengths, speakers = [], [], [], [], []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        waveform = torchaudio.load(row['file_path'])
        mfcc = mfcc_transform(waveform[0].to('cuda'))
        speaker = row['speaker']
        filename = row['file_path'][len('/workspace/data_pa/TIMIT/AUDIO_FILES/ORIGINAL/'):]

        indexes.append(index)
        mfcc_starts.append(mfcc_start)
        mfcc_lengths.append(mfcc.shape[2])
        speakers.append(speaker)

        mfcc_end = add_to_file(index, mfcc_start, mfcc.cpu(), speaker, filename,
                               mfcc_dataset, speaker_dataset, filepath_dataset,
                               meta_dataset, n_features=conf['data']['transform']['n_mfcc'], vox2=False)

        mfcc_ends.append(mfcc_end)
        mfcc_start = mfcc_end

    df = pd.DataFrame.from_dict({
        'MFCC_start': mfcc_starts,
        'MFCC_end': mfcc_ends,
        'MFCC_length': mfcc_lengths,
        'Speaker': speakers
    })

    df.to_csv(r'{}.csv'.format(df_name), index=False)


def create_h5_file_timit():
    conf = get_config()
    assert conf['data']['dataset'] == 'timit'

    df_base_path = Path('audio_datasets/dfs')
    train_fp = df_base_path / conf['data']['paths']['df']['train']
    val_fp = df_base_path / conf['data']['paths']['df']['val']
    test_fp = df_base_path / conf['data']['paths']['df']['test']

    _create_h5_file_timit(conf, train_fp, 'timit_mfcc_train.h5', 'timit_metadata_train')
    _create_h5_file_timit(conf, val_fp, 'timit_mfcc_valid.h5', 'timit_metadata_valid')
    _create_h5_file_timit(conf, test_fp, 'timit_mfcc_test.h5', 'timit_metadata_test')


def create_h5_file_vox2():
    conf = get_config()
    df_base_path = Path('audio_datasets/dfs')
    train_df = pd.read_csv(df_base_path / conf['data']['paths']['df']['train'])
    valid_df = pd.read_csv(df_base_path / conf['data']['paths']['df']['val'])
    test_df = pd.read_csv(df_base_path / conf['data']['paths']['df']['test'])

    train_files = train_df['file_path'].str.rstrip('.wav').str.strip('/workspace/data_pa/VOX2/AUDIO_FILES/').tolist()
    valid_files = valid_df['file_path'].str.rstrip('.wav').str.strip('/workspace/data_pa/VOX2/AUDIO_FILES/').tolist()
    test_files = test_df['file_path'].str.rstrip('.wav').str.strip('/workspace/data_pa/VOX2/AUDIO_FILES/').tolist()
    test_speakers = test_df['speaker'].tolist()
    test_speakers = list(dict.fromkeys(test_speakers))  # remove duplicates

    f = h5py.File('/workspace/data_pa/vox2_original.h5', 'r', libver='latest', swmr=True)
    f_train_new = h5py.File('/workspace/data_pa/vox2_mfcc_train.h5', 'w', libver='latest')
    f_valid_new = h5py.File('/workspace/data_pa/vox2_mfcc_valid.h5', 'w', libver='latest')
    f_test_new = h5py.File('/workspace/data_pa/vox2_mfcc_test.h5', 'w', libver='latest')

    number_of_entries = 0
    for speaker in tqdm(f['audio_names'].keys()):
        number_of_entries += len(f['audio_names'][speaker])

    print("Number of entries: {}".format(number_of_entries))

    mfcc_transformer = get_mfcc_transform(conf)

    mfcc_start_train, mfcc_start_valid, mfcc_start_test = 0, 0, 0
    i_train, i_valid, i_test = 0, 0, 0
    mfcc_datasets, speaker_datasets, filepath_datasets, meta_datasets = [], [], [], []
    for f_new in [f_train_new, f_valid_new, f_test_new]:
        mfcc_datasets.append(f_new.create_dataset('MFCC', (1, conf['data']['transform']['n_mfcc'], 1),
                                                  maxshape=(1, conf['data']['transform']['n_mfcc'], None), chunks=True,
                                                  dtype='float32'))
        speaker_datasets.append(f_new.create_dataset('Speaker', (number_of_entries, 1), 'S7'))
        filepath_datasets.append(f_new.create_dataset('Filepath', (number_of_entries, 1), 'S30'))
        meta_datasets.append(f_new.create_dataset('META', (number_of_entries, 2), dtype='int64'))

    for speaker in tqdm(f['audio_names'].keys()):
        for i in range(len(f['data'][speaker])):
            waveform = f['data'][speaker][i].reshape(1, f['statistics'][speaker][i])
            filename = f['audio_names'][speaker][i].decode('ascii')
            mfcc = mfcc_transformer(torch.from_numpy(waveform)).numpy()

            fp = speaker + "/" + filename.split('.m4a')[0]

            if fp in test_files:
                mfcc_start_test = add_to_file(i_test, mfcc_start_test, mfcc, speaker, filename,
                                              mfcc_datasets[2], speaker_datasets[2], filepath_datasets[2],
                                              meta_datasets[2], n_features=conf['data']['transform']['n_mfcc'])
                i_test += 1
            elif fp in valid_files:
                mfcc_start_valid = add_to_file(i_valid, mfcc_start_valid, mfcc, speaker, filename,
                                               mfcc_datasets[1], speaker_datasets[1], filepath_datasets[1],
                                               meta_datasets[1], n_features=conf['data']['transform']['n_mfcc'])
                i_valid += 1
            elif fp in train_files:
                mfcc_start_train = add_to_file(i_train, mfcc_start_train, mfcc, speaker, filename,
                                               mfcc_datasets[0], speaker_datasets[0], filepath_datasets[0],
                                               meta_datasets[0], n_features=conf['data']['transform']['n_mfcc'])
                i_train += 1
            else:
                logger.debug("File {} not defined in df".format(fp))

                if speaker in test_speakers:
                    mfcc_start_test = add_to_file(i_test, mfcc_start_test, mfcc, speaker, filename,
                                                  mfcc_datasets[2], speaker_datasets[2], filepath_datasets[2],
                                                  meta_datasets[2], n_features=conf['data']['transform']['n_mfcc'])
                    i_test += 1
                else:
                    mfcc_start_train = add_to_file(i_train, mfcc_start_train, mfcc, speaker, filename,
                                                   mfcc_datasets[0], speaker_datasets[0], filepath_datasets[0],
                                                   meta_datasets[0], n_features=conf['data']['transform']['n_mfcc'])
                    i_train += 1

    f.close()
    f_test_new.close()
    f_train_new.close()
    f_valid_new.close()
