import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import torchaudio
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt


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
    print("Total {}/{}".format(len(df_train)+len(df_val)+len(df_test), 2256492/2))

if __name__ == '__main__':
    create_df_vox2()
