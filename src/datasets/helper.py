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


def add_to_file(conf, i, mfcc_start, mfcc, speaker, filename, mfcc_d, speaker_d, filepath_d, meta_d):
    mfcc_end = mfcc_start + mfcc.shape[2]
    mfcc_d.resize((1, conf['data']['transform']['n_mfcc'], mfcc_end))
    mfcc_d[:, :, mfcc_start:mfcc_end] = mfcc
    speaker_d[i, :] = speaker
    filepath_d[i, :] = speaker + "/" + filename
    meta_d[i, :] = np.array([mfcc_start, mfcc_end])
    return mfcc_end


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


if __name__ == '__main__':
    os.chdir('../')
    create_h5_file_vox2()
    # conf = get_config()
    # from datasets.dataset import AudioDataset
    # from torch.utils.data import Subset
    #
    # conf['train']['batch_size'] = 1
    # train_loader, valid_loader, test_loader = get_loaders(conf, 'cuda')
    # train_loader.collate_fn = collate_fn_h5
    # f = h5py.File('vox2-train.h5', 'r')
    #
    # it = iter(train_loader)
    #
    # for i in range(20):
    #     mfcc, sizes, speaker = next(it)
    #     mfcc_start, mfcc_end = f['META'][i]
    #     mfcc2 = f['MFCC'][:, :, mfcc_start:mfcc_end]
    #     # waveform2 = f['Waveform'][:, waveform_start:waveform_end]
    #     speaker2 = f['Speaker'][i]
    #     assert np.all(mfcc2 == mfcc)
    #     # assert np.all(waveform2 == waveform.numpy())
    #     assert speaker2[0].decode('ascii') == speaker
