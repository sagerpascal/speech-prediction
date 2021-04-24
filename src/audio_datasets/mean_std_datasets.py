import numpy as np
from tqdm import tqdm
import pandas as pd
import torchaudio
from audio_datasets.preprocessing import get_mel_spectro_transform
from utils.meter import AverageValueMeter
from utils.conf_reader import get_config
import sys

conf = get_config()
df = pd.read_csv('dfs/libri-speech-orig-train.csv')
lengths = []
transform = get_mel_spectro_transform(conf).to('cpu')
to_db = torchaudio.transforms.AmplitudeToDB()

mean_meter, std_meter = AverageValueMeter(), AverageValueMeter()

with tqdm(range(len(df)), file=sys.stdout) as iterator:
    for i in iterator:
        file_pth = df['file_path'][i]
        waveform = torchaudio.load(file_pth)[0]
        mel_spectro = transform(waveform)
        mel_spectro = to_db(mel_spectro)
        mean, std = np.mean(mel_spectro.numpy()), np.std(mel_spectro.numpy())
        mean_meter.add(mean)
        std_meter.add(std)
        iterator.set_postfix_str("Avg. Mean: {}  Avg. Std: {}  Mean: {}  Std: {}".format(mean_meter.mean, std_meter.mean, mean, std))
        lengths.append(mel_spectro.shape[2])

df['mel_spectro_length'] = lengths
df.to_csv('dfs/libri-speech-orig-train.csv', index=False)












