# Prediction of Mel-spectrogram Sequences

A deep learning model to predict a subsequent sequence of a Mel-spectrogram.

Also have a look at [https://sagerpascal.github.io/speech-prediction/](https://sagerpascal.github.io/speech-prediction/)

## Setup Environment
The code can be executed in a Docker container. More detailed information about building the container can be found at [docker/README.md](docker/README.md)

After building the image, the container can be run using
````shell
nvidia-docker run -d speech-prediction
````

If an SSH connection shall be created (e.g. for deploying the code or for using the container as a remote debugger) add the option `-p <local-port>:22`:
````shell
nvidia-docker run -d -p 8888:22 speech-prediction
````

Inside the container, the TIMIT audio files are expected to be stored in the following folder:
````
/worspace/data_pa/TIMIT/AUDIO_FILES/ORIGINAL/train/<speaker-id>/
/worspace/data_pa/TIMIT/AUDIO_FILES/ORIGINAL/test/<speaker-id>/
````

For a better performance, `.h5` files instead of the raw audio files can be used. These files can be created with the helper-scripts
in the folder [audio_datasets](audio_datasets) and then be stored in the folder `/worspace/data_pa/TIMIT`.

Instead of saving these files inside the container, they can also be saved in an external folder. This external folder can then
be mapped inside the container using the command `-v <external-folder>:/workspace/data_pa`

````shell
nvidia-docker run -d -p 8888:22 -v <external-folder>:/workspace/data_pa speech-prediction
````

## Run Training or Evaluation

First, define the base config at [configs/config.yaml](configs/config.yaml), the either run training or validation with the command

````shell
python main.py [-h] [--mode MODE] [--lr LR] [--weight_decay WEIGHT_DECAY]
               [--load_weights LOAD_WEIGHTS] [--batch_size BATCH_SIZE]
               [--step_size STEP_SIZE] [--gamma GAMMA]
               [--gru_num_prenet_layer GRU_NUM_PRENET_LAYER]
               [--gru_num_rnn_layer GRU_NUM_RNN_LAYER]
               [--gru_num_postnet_layer GRU_NUM_POSTNET_LAYER]

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           'train' or 'eval'
  --lr LR               The learning rate
  --weight_decay WEIGHT_DECAY
                        Weight decay of the optimizer
  --load_weights LOAD_WEIGHTS
                        name of the model to load
  --batch_size BATCH_SIZE
                        The mini-batch size
  --step_size STEP_SIZE
                        LR scheduler step size
  --gamma GAMMA         LR scheduler gamma
  --gru_num_prenet_layer GRU_NUM_PRENET_LAYER
                        Number of prenet layers
  --gru_num_rnn_layer GRU_NUM_RNN_LAYER
                        Number of RNN layers
  --gru_num_postnet_layer GRU_NUM_POSTNET_LAYER
                        Number of postnet layers
````



