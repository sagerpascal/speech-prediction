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
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            filepath = Path(subdir) / filename
            data['file_name'].append(filename)
            data['file_path'].append(filepath)

    return data


def _create_df_file(path):
    files = _get_all_files(path)
    df = pd.DataFrame.from_dict(files)
    df['speaker'] = df['file_name'].str.replace('.WAV', '')
    print(df.head())
    return df


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



def create_df_timit():
    df_train = _create_df_file('/workspace/audio_data/TIMIT/AUDIO_FILES/ORIGINAL/train')
    df_test = _create_df_file('/workspace/audio_data/TIMIT/AUDIO_FILES/ORIGINAL/test')

    df_train, df_val = train_test_split(df_train, test_size=0.15)

    df_train.to_csv(r'{}.csv'.format('timit-orig-train'), index=False)
    df_val.to_csv(r'{}.csv'.format('timit-orig-val'), index=False)
    df_test.to_csv(r'{}.csv'.format('timit-orig-test'), index=False)


if __name__ == '__main__':
    analyze_timit_wav_length()
