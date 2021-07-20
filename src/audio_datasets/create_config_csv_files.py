import logging
import os
from pathlib import Path

import pandas as pd
import torchaudio
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from audio_datasets.preprocessing import get_mfcc_transform
from utils.conf_reader import get_config

logger = logging.getLogger(__name__)


def _get_all_files(path):
    data = {}
    data['file_name'], data['file_path'] = [], []
    for root, dirs, files in os.walk(path):
        for filename in files:
            filepath = Path(root) / filename
            data['file_name'].append(filename)
            data['file_path'].append(str(filepath.absolute()))

    return data


def _get_wav_lengths(files):
    lengths = []
    for f in tqdm(files):
        file = torchaudio.load(f)
        assert file[1] == 16000
        lengths.append(file[0].shape[1] / file[1])

    return lengths


def _create_df_file_timit(path):
    files = _get_all_files(path)
    df = pd.DataFrame.from_dict(files)
    df['speaker'] = df['file_name'].str.replace('.WAV', '')
    print(df.head())
    return df


def create_df_timit():
    """ Create the Dataframe .csv files stored in config/TIMIT"""

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
    """ Create the Dataframe .csv files stored in config/VOX2"""

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


def _create_df_file_libri(path):
    files = _get_all_files(path)
    df = pd.DataFrame.from_dict(files)
    df = df[df['file_path'].str.endswith('.flac')]
    df['speaker'] = df['file_path'].str.split('\\').str.get(-2)
    print(df.head())
    return df


def create_df_libri():
    """ Create the Dataframe .csv files stored in config/LibriSpeech"""

    df_train = _create_df_file_libri('D:/Projekte/temporal-speech-context/data/LibriSpeech/train-clean-360')

    df_train, df_val = train_test_split(df_train, test_size=0.03)

    df_train.to_csv(r'{}.csv'.format('libri-speech-orig-train'), index=False)
    df_val.to_csv(r'{}.csv'.format('libri-speech-orig-val'), index=False)


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

    df_base_path = Path('audio_datasets/dfs')
    train_fp = df_base_path / conf['data']['paths']['df']['train']
    val_fp = df_base_path / conf['data']['paths']['df']['val']
    test_fp = df_base_path / conf['data']['paths']['df']['test']

    _add_lengths_df_timit(conf, train_fp)
    _add_lengths_df_timit(conf, val_fp)
    _add_lengths_df_timit(conf, test_fp)


if __name__ == '__main__':
    os.chdir('../')
    create_df_libri()
