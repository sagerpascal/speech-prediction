import os
import time
from pathlib import Path

import h5pickle
import h5py
import numpy as np
import pandas as pd
import scipy.io.wavfile
import torchaudio
from tqdm import tqdm

from utils.conf_reader import get_config


def add_to_file(i, signal, speaker, filename, signal_d, speaker_d, filepath_d, meta_d):
    signal_d[i, :signal.shape[1]] = signal.numpy()
    speaker_d[i, :] = speaker.encode("ascii")
    filepath_d[i, :] = filename.encode("ascii")
    meta_d[i, :] = np.array([signal.shape[1]])


def create_h5_file_timit(orig_fp, meta_fp, h5_name, n_samples):
    base_path = Path('audio_datasets/dfs')
    df = pd.read_csv(base_path / orig_fp)
    number_of_entries = len(df)

    f_h5 = h5py.File(h5_name, 'w', libver='latest')
    dataset = f_h5.create_dataset('RAW', (n_samples, 124621), chunks=True, dtype='float32')
    speaker_dataset = f_h5.create_dataset('Speaker', (number_of_entries, 1), 'S7')
    filepath_dataset = f_h5.create_dataset('Filepath', (number_of_entries, 1), 'S30')
    meta_dataset = f_h5.create_dataset('META', (number_of_entries, 1), dtype='int64')

    indexes, raw_lengths, speakers = [], [], []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        waveform = torchaudio.load(row['file_path'])
        signal = waveform[0]
        speaker = row['speaker']
        filename = row['file_path'][len('/workspace/data_pa/TIMIT/AUDIO_FILES/ORIGINAL/'):]

        indexes.append(index)
        raw_lengths.append(signal.shape[1])
        speakers.append(speaker)

        add_to_file(index, signal.cpu(), speaker, filename,
                    dataset, speaker_dataset, filepath_dataset,
                    meta_dataset)

    df_meta = pd.DataFrame.from_dict({
        'raw_length': raw_lengths,
        'Speaker': speakers
    })

    df_meta.to_csv(r'{}'.format(base_path / meta_fp), index=False)

    df['raw_length'] = raw_lengths
    df.to_csv(base_path / orig_fp, index=False)

    f_h5.close()

def calc_mean_std():
    h5_file = h5pickle.File('timit_raw_train.h5', 'r', skip_cache=False)
    avg_b, std_b = [], []  # do not calc the avg of all files to avoid memory leakage
    data = h5_file['RAW']
    buckets = 1000
    bucket_size = data.shape[1] // buckets
    for i in tqdm(range(buckets)):
        start = i * bucket_size
        end = (i + 1) * bucket_size
        avg_b.append(np.mean(data[:, start:end]))
        std_b.append(np.std(data[:, start:end]))
    print("RAW FILES: AVG={} STD={}".format(np.mean(np.array(avg_b)), np.std(np.array(std_b))))


def check_h5_files():
    base_path = Path('audio_datasets/dfs')
    metadata_df = pd.read_csv(base_path / 'timit_raw_metadata_train.csv')
    h5_file = h5pickle.File('timit_raw_train.h5', 'r', skip_cache=False)

    for i in range(5):
        start = metadata_df['raw_start'][i]
        end = metadata_df['raw_end'][i]
        waveform = h5_file['RAW'][:, start:end]
        scipy.io.wavfile.write('check/{}.wav'.format(i), 16000, waveform.T)


if __name__ == '__main__':
    os.chdir('../')
    conf = get_config()
    # check_h5_files()
    create_h5_file_timit('timit-orig-train.csv', 'timit_raw_metadata_train.csv',
                         'timit_raw_train.h5', 3927)
    create_h5_file_timit('timit-orig-val.csv', 'timit_raw_metadata_valid.csv',
                         'timit_raw_valid.h5', 693)
    create_h5_file_timit('timit-orig-test.csv', 'timit_raw_metadata_test.csv',
                         'timit_raw_test.h5', 1680)
    # calc_mean_std()

    # h5_file2 = h5pickle.File('timit_raw_valid.h5', 'r', skip_cache=False)
    #
    # t1 = time.time()
    # data = h5_file2['RAW'][692, :48640]  # size=204'917'204
    # t2 = time.time()
    #
    # h5_file = h5pickle.File('/workspace/data_pa/TIMIT/timit_raw_valid-128.h5', 'r', skip_cache=False)
    #
    # t3 = time.time()
    # data2 = h5_file['RAW'][:, 34060474:34109114]  # size=409'608'927
    # t4 = time.time()
    #
    # df1 = pd.read_csv('audio_datasets/dfs/timit_raw_metadata_valid-128.csv')
    # df2 = pd.read_csv('audio_datasets/dfs/timit_raw_metadata_valid.csv')
    #
    # for i in range(693):
    #     l1, s1, e1, = df1['raw_length'][i], df1['raw_start'][i], df1['raw_end'][i]
    #     l2 = df2['raw_length'][i]
    #     data = h5_file['RAW'][:, s1:e1]
    #     data2 = h5_file2['RAW'][i, :l2]
    #     assert np.all(data == data2)
    #
    # print("{}, {}".format(t2-t1, t4-t3))
