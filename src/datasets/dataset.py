import pandas as pd
from pathlib import Path
import torchaudio
from torch.utils.data import Dataset

# http://www.openslr.org/12/
# https://lionbridge.ai/datasets/best-speech-recognition-datasets-for-machine-learning/
# https://www.sciencedirect.com/science/article/pii/S0885230819302712


class AudioDataset(Dataset):

    def __init__(self, conf, mode, preprocess=None, df_base_path='datasets/dfs'):

        self.conf = conf
        self.preprocess = preprocess
        df_base_path = Path(df_base_path)

        if mode == 'train':
            self.fp = Path(conf['data']['paths']['base']) / conf['data']['paths']['train']
            df_fp = df_base_path / conf['data']['paths']['df']['train']
        elif mode == 'val':
            self.fp = Path(conf['data']['paths']['base']) / conf['data']['paths']['val']
            df_fp = df_base_path / conf['data']['paths']['df']['val']
        elif mode == 'test':
            self.fp = Path(conf['data']['paths']['base']) / conf['data']['paths']['test']
            df_fp = df_base_path / conf['data']['paths']['df']['test']
        else:
            raise AttributeError("Unknown mode: {}".format(mode))

        self.df = pd.read_csv(df_fp)

    def __getitem__(self, item):

        waveform = torchaudio.load(self.fp / self.df['file_path'][item])
        speaker = self.df['speaker'][item]

        if self.preprocess is not None:
            waveform, speaker = self.preprocess(waveform, speaker)

        return waveform[0], speaker

    def __len__(self):
        return len(self.df)
