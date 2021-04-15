import os
from pathlib import Path

import h5pickle
import h5py
import numpy as np
import pandas as pd
import scipy.io.wavfile
import torchaudio
from tqdm import tqdm

from utils.conf_reader import get_config


def add_to_file(i, signal_start, signal, speaker, filename, signal_d, speaker_d, filepath_d, meta_d):
    signal_end = signal_start + signal.shape[1]
    signal_d.resize((1, signal_end))
    signal_d[:, signal_start:signal_end] = signal
    speaker_d[i, :] = speaker.encode("ascii")
    filepath_d[i, :] = filename.encode("ascii")
    meta_d[i, :] = np.array([signal_start, signal_end])
    return signal_end


def create_h5_file_timit(orig_fp, meta_fp, h5_name):
    base_path = Path('audio_datasets/dfs')
    df = pd.read_csv(base_path / orig_fp)
    number_of_entries = len(df)

    f_h5 = h5py.File(h5_name, 'w', libver='latest')
    dataset = f_h5.create_dataset('RAW', (1, 1), maxshape=(1, None), chunks=True, dtype='float32')
    speaker_dataset = f_h5.create_dataset('Speaker', (number_of_entries, 1), 'S7')
    filepath_dataset = f_h5.create_dataset('Filepath', (number_of_entries, 1), 'S30')
    meta_dataset = f_h5.create_dataset('META', (number_of_entries, 2), dtype='int64')

    raw_start = 0
    indexes, raw_starts, raw_ends, raw_lengths, speakers = [], [], [], [], []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        waveform = torchaudio.load(row['file_path'])
        signal = waveform[0]
        speaker = row['speaker']
        filename = row['file_path'][len('/workspace/data_pa/TIMIT/AUDIO_FILES/ORIGINAL/'):]

        indexes.append(index)
        raw_starts.append(raw_start)
        raw_lengths.append(signal.shape[1])
        speakers.append(speaker)

        raw_end = add_to_file(index, raw_start, signal.cpu(), speaker, filename,
                              dataset, speaker_dataset, filepath_dataset,
                              meta_dataset)

        assert raw_end - raw_start == signal.shape[1]
        raw_ends.append(raw_end)
        raw_start = raw_end

    df_meta = pd.DataFrame.from_dict({
        'raw_start': raw_starts,
        'raw_end': raw_ends,
        'raw_length': raw_lengths,
        'Speaker': speakers
    })

    df_meta.to_csv(r'{}'.format(base_path / meta_fp), index=False)

    df['raw_length'] = raw_lengths
    df.to_csv(base_path / orig_fp, index=False)


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
    # create_h5_file_timit('timit-orig-train.csv', 'timit_raw_metadata_train.csv',
    #                      'timit_raw_train.h5')
    # create_h5_file_timit('timit-orig-val.csv', 'timit_raw_metadata_valid.csv',
    #                      'timit_raw_valid.h5')
    # create_h5_file_timit('timit-orig-test.csv', 'timit_raw_metadata_test.csv',
    #                      'timit_raw_test.h5')
    calc_mean_std()
