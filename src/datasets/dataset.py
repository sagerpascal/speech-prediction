import pandas as pd
from pathlib import Path
import torchaudio
from torch.utils.data import Dataset
from datasets.preprocessing import get_mfcc_transform, get_mfcc_preprocess_fn

# http://www.openslr.org/12/
# https://lionbridge.ai/datasets/best-speech-recognition-datasets-for-machine-learning/
# https://www.sciencedirect.com/science/article/pii/S0885230819302712


class AudioDataset(Dataset):

    def __init__(self, conf, mode, df_base_path='datasets/dfs'):

        self.conf = conf
        df_base_path = Path(df_base_path)

        if mode == 'train':
            df_fp = df_base_path / conf['data']['paths']['df']['train']
        elif mode == 'val':
            df_fp = df_base_path / conf['data']['paths']['df']['val']
        elif mode == 'test':
            df_fp = df_base_path / conf['data']['paths']['df']['test']
        else:
            raise AttributeError("Unknown mode: {}".format(mode))

        self.df = pd.read_csv(df_fp)
        self.mfcc_transform = get_mfcc_transform(conf).to('cuda')
        self.preprocess = get_mfcc_preprocess_fn(mask_pos=conf['masking']['position'],
                                                 n_frames=conf['masking']['n_frames'],
                                                 k_frames=conf['masking']['k_frames'],
                                                 use_random_pos=conf['masking']['use_random_pos'])

    def __getitem__(self, item):

        waveform = torchaudio.load(self.df['file_path'][item])
        speaker = self.df['speaker'][item]

        mfcc = self.mfcc_transform(waveform[0])
        data, target = self.preprocess(mfcc)

        return data, target, mfcc, waveform[0], speaker

    def __len__(self):
        return len(self.df)
