import os

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import torchaudio
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils.conf_reader import get_config
import h5py
from dataloader import get_loaders
import copy
from datasets.preprocessing import get_mfcc_transform
import torch
import logging

logger = logging.getLogger(__name__)

def _get_all_files(path):
    data = {}
    data['file_name'], data['file_path'] = [], []
    for root, dirs, files in os.walk(path):
        for filename in files:
            filepath = Path(root) / filename
            data['file_name'].append(filename)
            data['file_path'].append(filepath)

    return data


def _get_wav_lengths(files):
    lengths = []
    for f in tqdm(files):
        file = torchaudio.load(f)
        assert file[1] == 16000
        lengths.append(file[0].shape[1] / file[1])

    return lengths


def analyze_timit_wav_length():
    df_fp = Path('dfs')
    df_train = pd.read_csv(df_fp / 'timit-orig-train.csv')
    df_val = pd.read_csv(df_fp / 'timit-orig-val.csv')
    df_test = pd.read_csv(df_fp / 'timit-orig-test.csv')

    file_lengths_train = _get_wav_lengths(df_train['file_path'])
    file_lengths_val = _get_wav_lengths(df_val['file_path'])
    file_lengths_test = _get_wav_lengths(df_test['file_path'])

    df = pd.DataFrame.from_dict({
        'frequency per duration [s]': file_lengths_train + file_lengths_val + file_lengths_test,
    })

    ax = df.hist(bins=18)
    plt.savefig("timit-length-dist.png")


def _create_df_file_timit(path):
    files = _get_all_files(path)
    df = pd.DataFrame.from_dict(files)
    df['speaker'] = df['file_name'].str.replace('.WAV', '')
    print(df.head())
    return df


def create_df_timit():
    df_train = _create_df_file_timit('/workspace/audio_data/TIMIT/AUDIO_FILES/ORIGINAL/train')
    df_test = _create_df_file_timit('/workspace/audio_data/TIMIT/AUDIO_FILES/ORIGINAL/test')

    df_train, df_val = train_test_split(df_train, test_size=0.15)

    df_train.to_csv(r'{}.csv'.format('timit-orig-train'), index=False)
    df_val.to_csv(r'{}.csv'.format('timit-orig-val'), index=False)
    df_test.to_csv(r'{}.csv'.format('timit-orig-test'), index=False)


def _create_df_file_vox2(metadata):
    base_path = '/workspace/data_pa/VOX2/AUDIO_FILES/'
    del metadata['Set']  # delete the Set column
    metadata.rename(columns={'VoxCeleb2ID': 'speaker', 'VGGFace2ID': 'face_id', 'Gender': 'gender'}, inplace=True)

    dataset = {
        'speaker': [],
        'face_id': [],
        'gender': [],
        'file_path': [],
    }
    for index, row in metadata.iterrows():
        files = _get_all_files(Path(base_path) / row['speaker'])
        for fpath in files['file_path']:
            if str(fpath).endswith('.wav'):
                dataset['speaker'].append(row['speaker'])
                dataset['face_id'].append(row['face_id'])
                dataset['gender'].append(row['gender'])
                dataset['file_path'].append(fpath)
    return pd.DataFrame.from_dict(dataset)


def create_df_vox2():
    meta = pd.read_csv(Path('dfs') / 'vox2_meta.csv')
    print(meta.head())
    meta_dev = meta[meta['Set'] == 'dev']
    meta_test = meta[meta['Set'] == 'test']

    meta_train, meta_val = train_test_split(meta_dev, test_size=0.01)

    df_train = _create_df_file_vox2(meta_train)
    df_val = _create_df_file_vox2(meta_val)
    df_test = _create_df_file_vox2(meta_test)

    df_train.to_csv(r'{}.csv'.format('vox2-orig-train'), index=False)
    df_val.to_csv(r'{}.csv'.format('vox2-orig-val'), index=False)
    df_test.to_csv(r'{}.csv'.format('vox2-orig-test'), index=False)

    print("File lengths:\nTrain: {}\nValid: {}\nTest: {}".format(len(df_train), len(df_val), len(df_test)))
    print("Total {}/{}".format(len(df_train) + len(df_val) + len(df_test), 2256492 / 2))


def add_to_file(conf, i, mfcc_start, mfcc, speaker, filename, mfcc_d, speaker_d, filepath_d, meta_d, vox2=True):
    mfcc_end = mfcc_start + mfcc.shape[2]
    mfcc_d.resize((1, conf['data']['transform']['n_mfcc'], mfcc_end))
    mfcc_d[:, :, mfcc_start:mfcc_end] = mfcc
    speaker_d[i, :] = speaker
    filepath_d[i, :] = speaker + "/" + filename if vox2 else filename
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

        mfcc_end = add_to_file(conf, index, mfcc_start, mfcc.cpu(), speaker, filename,
                                      mfcc_dataset, speaker_dataset, filepath_dataset,
                                      meta_dataset, vox2=False)

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

    df_base_path = Path('datasets/dfs')
    train_fp = df_base_path / conf['data']['paths']['df']['train']
    val_fp = df_base_path / conf['data']['paths']['df']['val']
    test_fp = df_base_path / conf['data']['paths']['df']['test']

    _create_h5_file_timit(conf, train_fp, 'timit_mfcc_train.h5', 'timit_metadata_train')
    _create_h5_file_timit(conf, val_fp, 'timit_mfcc_valid.h5', 'timit_metadata_valid')
    _create_h5_file_timit(conf, test_fp, 'timit_mfcc_test.h5', 'timit_metadata_test')



def create_h5_file_vox2():
    conf = get_config()
    df_base_path = Path('datasets/dfs')
    train_df = pd.read_csv(df_base_path / conf['data']['paths']['df']['train'])
    valid_df = pd.read_csv(df_base_path / conf['data']['paths']['df']['val'])
    test_df = pd.read_csv(df_base_path / conf['data']['paths']['df']['test'])

    train_files = train_df['file_path'].str.rstrip('.wav').str.strip('/workspace/data_pa/VOX2/AUDIO_FILES/').tolist()
    valid_files = valid_df['file_path'].str.rstrip('.wav').str.strip('/workspace/data_pa/VOX2/AUDIO_FILES/').tolist()
    test_files = test_df['file_path'].str.rstrip('.wav').str.strip('/workspace/data_pa/VOX2/AUDIO_FILES/').tolist()
    test_speakers = test_df['speaker'].tolist()
    test_speakers = list(dict.fromkeys(test_speakers)) # remove duplicates

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
                mfcc_start_test = add_to_file(conf, i_test, mfcc_start_test, mfcc, speaker, filename,
                                              mfcc_datasets[2], speaker_datasets[2], filepath_datasets[2],
                                              meta_datasets[2])
                i_test += 1
            elif fp in valid_files:
                mfcc_start_valid = add_to_file(conf, i_valid, mfcc_start_valid, mfcc, speaker, filename,
                                               mfcc_datasets[1], speaker_datasets[1], filepath_datasets[1],
                                               meta_datasets[1])
                i_valid += 1
            elif fp in train_files:
                mfcc_start_train = add_to_file(conf, i_train, mfcc_start_train, mfcc, speaker, filename,
                                               mfcc_datasets[0], speaker_datasets[0], filepath_datasets[0],
                                               meta_datasets[0])
                i_train += 1
            else:
                logger.debug("File {} not defined in df".format(fp))

                if speaker in test_speakers:
                    mfcc_start_test = add_to_file(conf, i_test, mfcc_start_test, mfcc, speaker, filename,
                                                  mfcc_datasets[2], speaker_datasets[2], filepath_datasets[2],
                                                  meta_datasets[2])
                    i_test += 1
                else:
                    mfcc_start_train = add_to_file(conf, i_train, mfcc_start_train, mfcc, speaker, filename,
                                                   mfcc_datasets[0], speaker_datasets[0], filepath_datasets[0],
                                                   meta_datasets[0])
                    i_train += 1

    f.close()
    f_test_new.close()
    f_train_new.close()
    f_valid_new.close()


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


def _add_lengths_df_timit(conf, df_fp):
    df = pd.read_csv(df_fp)
    mfcc_transform = get_mfcc_transform(conf).to('cuda')

    indexes, mfcc_lengths = [], []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        waveform = torchaudio.load(row['file_path'])
        mfcc = mfcc_transform(waveform[0].to('cuda'))
        mfcc_lengths.append(mfcc.shape[2])
        indexes.append(index)

    df['MFCC_length'] = mfcc_lengths
    df.to_csv(df_fp, index=False)


def create_metadata_timit():
    conf = get_config()
    assert conf['data']['dataset'] == 'timit'

    df_base_path = Path('datasets/dfs')
    train_fp = df_base_path / conf['data']['paths']['df']['train']
    val_fp = df_base_path / conf['data']['paths']['df']['val']
    test_fp = df_base_path / conf['data']['paths']['df']['test']

    _add_lengths_df_timit(conf, train_fp)
    _add_lengths_df_timit(conf, val_fp)
    _add_lengths_df_timit(conf, test_fp)

if __name__ == '__main__':
    os.chdir('../')
    create_h5_file_timit()
