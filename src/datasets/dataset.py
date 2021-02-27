import numpy as np
import h5py
import os

def make_accessible(path):
    os.popen(f'chmod a=rwx -R "{path}"').read()

def get_destination_path(dataset, subset, config):
    mass = config['GLOBAL']['MASS_DATASET_PATH']
    if mass[-1] != '/':
        mass = f'{mass}/'
    quick = config['GLOBAL']['DATASET_PATH']
    if quick[-1] != '/':
        quick = f'{quick}/'

    destination_path = f'{dataset}/04_PROCESSED_DATASETS/{config["TRANSFORMATION"]["TYPE"]}/'
    for key in ['PRE_EMPHASIS', 'NFFT', 'WINDOW', 'FRAME_LENGTH', 'FRAME_STEP', 'N_MELS', 'FMIN', 'FMAX', 'MFCC_RANGE']:
        if key in config['TRANSFORMATION']:
            v = config['TRANSFORMATION'][key]
            if type(v) is list:
                v = '-'.join([f'{x}' for x in v])
            destination_path = f'{destination_path}{key}={v};'

    destination_path = destination_path[:-1]
    file_path = f'{destination_path}/{subset}.h5'

    os.makedirs(f'{mass}{destination_path}', exist_ok=True)
    os.makedirs(f'{quick}{destination_path}', exist_ok=True)

    make_accessible(f'{mass}{dataset}/04_PROCESSED_DATASETS')
    make_accessible(f'{quick}{dataset}/04_PROCESSED_DATASETS')

    mass = (f'{mass}{file_path}', os.path.isfile(f'{mass}{file_path}'))
    quick = (f'{quick}{file_path}', os.path.isfile(f'{quick}{file_path}'))
    return mass, quick

class Dataset:

    def __init__(self, conf):
        self.conf = conf
        self.h5_files = []

    def load_train_val_locs(self):
        dataset = self.conf['data']['dataset']
        subset = self.conf['data']['subset']
        train_audio_list = self.conf['data']['audio_list_train']
        val_audio_list = self.conf['data']['audio_list_val']
        dataset_path = self.conf['data']['dataset_path']

        train_audios = np.loadtxt(f'{dataset_path}{dataset}/02_AUDIONAME_LISTS/{train_audio_list}.txt', str)
        val_audios = np.loadtxt(f'{dataset_path}{dataset}/02_AUDIONAME_LISTS/{val_audio_list}.txt', str)

        h5_file = get_destination_path(dataset, subset, self.conf)[1][0]
        if h5_file not in self.h5_files:
            self.h5_files.append(h5_file)
        dataset_id = self.h5_files.index(h5_file)

        # Load Metadata from H5 File
        with h5py.File(h5_file, mode='r') as src:
            self.conf['data']['num_freqs'] = src['data'].shape[-1]

            # Load References into RAM
            audio_refs = np.array(src['META/AUDIOS'][:], dtype=str)
            sample_refs = src['META/LOCS'][:]

            # Allow only samples that are longer than config['DATA']['SEGMENT_LENGTH']
            sample_refs = sample_refs[sample_refs[:, 3] - sample_refs[:, 2] >= self.conf['data']['SEGMENT_LENGTH']]

        # Select Sample References of audios in config['TRAINING']['AUDIO_LIST_TRAIN']
        train_audio_idxs = np.in1d(audio_refs, train_audios).nonzero()[0]
        train_sample_idxs = np.in1d(sample_refs[:, 1], train_audio_idxs).nonzero()[0]
        train_sample_locs = sample_refs[train_sample_idxs]
        train_dataset_ids = (np.ones(train_sample_locs.shape[0]) * dataset_id).reshape((-1, 1))

        # Select Sample References of audios in config['TRAINING']['AUDIO_LIST_VAL']
        val_audio_idxs = np.in1d(audio_refs, val_audios).nonzero()[0]
        val_sample_idxs = np.in1d(sample_refs[:, 1], val_audio_idxs).nonzero()[0]
        val_sample_locs = sample_refs[val_sample_idxs]
        val_dataset_ids = (np.ones(val_sample_locs.shape[0]) * dataset_id).reshape((-1, 1))

        # Remap Speaker References to One-Hot Bins
        speaker_idxs = np.concatenate((train_sample_locs[:, 0], val_sample_locs[:, 0]))
        self.speaker_map, speaker_refs = np.unique(speaker_idxs, return_inverse=True)
        speaker_refs = speaker_refs.reshape((-1, 1))

        split_point = train_sample_locs.shape[0]
        self.config['TRAINING']['NUM_SPEAKERS'] = len(self.speaker_map)

        # Generate List of Audio Label & Sample Reference Pairs
        train_speaker_refs = speaker_refs[:split_point]
        self.train_sample_locs = np.concatenate((train_dataset_ids, train_speaker_refs, train_sample_locs[:, 2:4]),
                                                axis=1)
        self.train_steps = int(np.floor(len(self.train_sample_locs) / self.config['TRAINING']['BATCH_SIZE']))

        val_speaker_refs = speaker_refs[split_point:]
        self.val_sample_locs = np.concatenate((val_dataset_ids, val_speaker_refs, val_sample_locs[:, 2:4]), axis=1)
        self.val_steps = int(np.floor(len(self.val_sample_locs) / self.config['TRAINING']['BATCH_SIZE']))


