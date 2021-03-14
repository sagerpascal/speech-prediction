import h5pickle
import torch
from pathlib import Path
from torch.utils.data import Dataset
from datasets.preprocessing import get_mfcc_preprocess_fn


class AudioDatasetH5(Dataset):

    def __init__(self, conf, mode, df_base_path='/workspace/data_pa/'):

        df_base_path = Path(df_base_path)

        if mode == 'train':
            h5_fp = df_base_path / conf['data']['paths']['h5']['train']
        elif mode == 'val':
            h5_fp = df_base_path / conf['data']['paths']['h5']['val']
        elif mode == 'test':
            h5_fp = df_base_path / conf['data']['paths']['h5']['test']
        else:
            raise AttributeError("Unknown mode: {}".format(mode))

        self.h5_file = h5pickle.File(str(h5_fp.resolve()), 'r', skip_cache=False) #, libver='latest', swmr=True)
        self.preprocess = get_mfcc_preprocess_fn(mask_pos=conf['masking']['position'],
                                                 n_frames=conf['masking']['n_frames'],
                                                 k_frames=conf['masking']['k_frames'],
                                                 use_random_pos=conf['masking']['use_random_pos'])
        self.number_of_files = len(self.h5_file['META'])
        self.k_frames = conf['masking']['k_frames']

    def __getitem__(self, item):

        mfcc_start, mfcc_end = self.h5_file['META'][item]
        mfcc = self.h5_file['MFCC'][:, :, mfcc_start:mfcc_end]
        speaker = self.h5_file['Speaker'][item][0].decode('ascii')
        # filepath = self.h5_file['Filepath'][item][0].decode('ascii')

        mfcc = torch.from_numpy(mfcc)

        data, target = self.preprocess(mfcc)

        return data, target, mfcc, None, speaker

    def __len__(self):
        return self.number_of_files

    def __del__(self):
        self.h5_file.close()
