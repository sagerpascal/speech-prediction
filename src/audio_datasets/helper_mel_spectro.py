import os
from pathlib import Path

import h5py
import pandas as pd
import torchaudio
from tqdm import tqdm
import numpy as np

from audio_datasets.helper_mfcc import add_to_file
from audio_datasets.preprocessing import get_mel_spectro_transform
from utils.conf_reader import get_config


def create_h5_file(conf, orig_fp, meta_fp, h5_name):
    base_path = Path('audio_datasets/dfs')
    df = pd.read_csv(base_path / orig_fp)
    mel_spectro_transform = get_mel_spectro_transform(conf).to('cuda')
    to_db = torchaudio.transforms.AmplitudeToDB()
    number_of_entries = len(df)

    f_h5 = h5py.File(h5_name, 'w', libver='latest')
    mel_spectro_dataset = f_h5.create_dataset('Mel-Spectrogram', (1, conf['data']['transform']['n_mels'], 1),
                                              maxshape=(1, conf['data']['transform']['n_mels'], None), chunks=True,
                                              dtype='float32')
    speaker_dataset = f_h5.create_dataset('Speaker', (number_of_entries, 1), 'S7')
    filepath_dataset = f_h5.create_dataset('Filepath', (number_of_entries, 1), 'S30')
    meta_dataset = f_h5.create_dataset('META', (number_of_entries, 2), dtype='int64')

    mel_spectro_start = 0
    indexes, mel_spectro_starts, mel_spectro_ends, mel_spectro_lengths, speakers = [], [], [], [], []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        waveform = torchaudio.load(row['file_path'])
        mel_spectro = mel_spectro_transform(waveform[0].to('cuda'))
        mel_spectro = to_db(mel_spectro)

        speaker = str(row['speaker'])
        filename = row['file_path'][len('/workspace/data_pa/TIMIT/AUDIO_FILES/ORIGINAL/'):]

        indexes.append(index)
        mel_spectro_starts.append(mel_spectro_start)
        mel_spectro_lengths.append(mel_spectro.shape[2])
        speakers.append(speaker)

        mel_spectro_end = add_to_file(index, mel_spectro_start, mel_spectro.cpu(), speaker, filename,
                                      mel_spectro_dataset, speaker_dataset, filepath_dataset,
                                      meta_dataset, n_features=conf['data']['transform']['n_mels'], vox2=False)

        mel_spectro_ends.append(mel_spectro_end)
        mel_spectro_start = mel_spectro_end

    df_meta = pd.DataFrame.from_dict({
        'mel_spectro_start': mel_spectro_starts,
        'mel_spectro_end': mel_spectro_ends,
        'mel_spectro_length': mel_spectro_lengths,
        'Speaker': speakers
    })

    df_meta.to_csv(r'{}'.format(base_path / meta_fp), index=False)

    df['mel_spectro_length'] = mel_spectro_lengths
    df.to_csv(base_path / orig_fp, index=False)


def calc_mean_std(fp):
    h5_file = h5py.File(fp, 'r')
    mel_spectros = h5_file['Mel-Spectrogram']
    mean_l, std_l = [], []
    n_buckets = 50
    bucket_size = mel_spectros.shape[2] // n_buckets
    for i in tqdm(range(n_buckets)):
        start_idx, end_idx = i*bucket_size, (i+1)*bucket_size
        mel_spectros_np = mel_spectros[:, :, start_idx:end_idx]
        mean_l.append(np.mean(mel_spectros_np))
        std_l.append(np.std(mel_spectros_np))
    print("Mel-Spctrogram: AVG={} STD={}".format(np.mean(np.array(mean_l)), np.std(np.array(std_l))))



if __name__ == '__main__':
    os.chdir('../')
    conf = get_config()
    # create_h5_file(conf, 'timit-orig-train.csv', 'timit_mel_spectro_metadata_train_dB-80.csv',
    #                'timit_mel-spectro_train_dB-80.h5')
    # create_h5_file(conf, 'timit-orig-val.csv', 'timit_mel_spectro_metadata_valid_dB-80.csv',
    #                'timit_mel-spectro_valid_dB-80.h5')
    # create_h5_file(conf, 'timit-orig-test.csv', 'timit_mel_spectro_metadata_test_dB-80.csv',
    #                'timit_mel-spectro_test_dB-80.h5')
    # calc_mean_std('timit_mel-spectro_train_dB-80.h5')

    # create_h5_file(conf, 'libri-speech-orig-train.csv', 'libri-speech_mel_spectro_metadata_train.csv',
    #                'libri-speech_mel-spectro_train_dB-80.h5')
    # create_h5_file(conf, 'libri-speech-orig-val.csv', 'libri-speech_mel_spectro_metadata_val.csv',
    #                    'libri-speech_mel-spectro_val_dB-80.h5')
    calc_mean_std('D:/Projekte/temporal-speech-context/data/libri-speech_mel-spectro_train_dB-80.h5')
