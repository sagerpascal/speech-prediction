import h5pickle
import torch
from pathlib import Path
from torch.utils.data import Dataset
from datasets.preprocessing import get_mfcc_preprocess_fn
import pandas as pd


class AudioDatasetH5(Dataset):

    def __init__(self, conf, mode, df_base_path='/workspace/data_pa/'):

        df_base_path = Path(df_base_path)
        meta_base_path = Path('datasets/dfs')

        if mode == 'train':
            h5_fp = df_base_path / conf['data']['paths']['h5']['train']
            md_fp = meta_base_path / conf['data']['paths']['h5']['metadata']['train']
        elif mode == 'val':
            h5_fp = df_base_path / conf['data']['paths']['h5']['val']
            md_fp = meta_base_path / conf['data']['paths']['h5']['metadata']['val']
        elif mode == 'test':
            h5_fp = df_base_path / conf['data']['paths']['h5']['test']
            md_fp = meta_base_path / conf['data']['paths']['h5']['metadata']['test']
        else:
            raise AttributeError("Unknown mode: {}".format(mode))

        self.metadata_df = pd.read_csv(md_fp)
        self.h5_file = h5pickle.File(str(h5_fp.resolve()), 'r', skip_cache=False) #, libver='latest', swmr=True)
        self.preprocess = get_mfcc_preprocess_fn(mask_pos=conf['masking']['position'],
                                                 n_frames=conf['masking']['n_frames'],
                                                 k_frames=conf['masking']['k_frames'],
                                                 use_random_pos=conf['masking']['use_random_pos'])

        self.k_frames = conf['masking']['k_frames']
        self.n_frames = conf['masking']['n_frames']

        # ignore all files < k_frames + n_frames
        self.metadata_df = self.metadata_df[self.metadata_df['MFCC_length'] >= (self.n_frames + self.k_frames)]
        print("{} set has {} valid entries".format(mode, len(self.metadata_df)))

    def __getitem__(self, item):

        mfcc_start = self.metadata_df['MFCC_start'][item]
        mfcc_end = self.metadata_df['MFCC_end'][item]
        speaker = self.metadata_df['Speaker'][item]

        mfcc = self.h5_file['MFCC'][:, :, mfcc_start:mfcc_end]
        # mfcc_start, mfcc_end = self.h5_file['META'][item]
        # speaker = self.h5_file['Speaker'][item][0].decode('ascii')
        # filepath = self.h5_file['Filepath'][item][0].decode('ascii')

        mfcc = torch.from_numpy(mfcc)

        data, target = self.preprocess(mfcc)

        return data, target, mfcc, None, speaker

    def __len__(self):
        return len(self.metadata_df)

    def __del__(self):
        self.h5_file.close()
