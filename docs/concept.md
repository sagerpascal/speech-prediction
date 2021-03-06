---
layout: default
title: Concept
nav_order: 1
---

# Concept

As described in the [problem definition]({{ site.baseurl }}{% link index.md %}), the audio file is converted to a sequence
of MFCC frames. To do so, the audio files are first loaded by the dataloader. Then, the function `torchaudio.transforms.MFCC`
is used to calculate the MFCC frames. This function first calculates the MelSpectrogram (composition of Spectrogram and
MelScale) and then creates the Mel-frequency cepstrum coefficients. For the calculation, the following parameters are 
used:

> TODO: Could be better if not the whole audio file is loaded to calculate the MFCC (use only a part of it). Otherwise a batch is biased towards one speaker?!

- `sample_rate` (Sample rate of audio signal): $$16000$$
- `n_mfcc` (Number of mfcc coefficients to retain): $$20$$ or $$40$$
- `win_length` (Window size): $$16000 * 0.03 = 400$$
- `hop_length` (Length of hop between STFT windows): `win_length // 2` 
- `n_fft` (Size of FFT, creates `n_fft // 2 + 1` bins): $$512$$
- `f_min` (Minimum frequency for the Mel-Spectrogram): $$0$$
- `f_max` (Maximum frequency for the Mel-Spectrogram): $$8000$$


Per second, this results in

$$
\text{FPS} = \text{sr} / \frac{\text{n_fft}}{2} = 16000 / \frac{400}{2} = 80
$$

frames. The goal is then to predict the $k$ frames based on $n$ given frames. Various experiments could be interesting:
- $n$ frames are given, calculate the $k$ subsequent frames
- $n$ frames are given, calculate the $k$ previous frames
- for both experiments: 
  - start with a large $n$ and decrease it,
  - and/or start with a small $k$ and increase it




For the network, the $n$ given frames are the input data `x`, and the $k$ frames to be predicted are the label `y`. For
a first baseline, a simple Transfomer network is used. Currently, no literature exists comparing MFCC frame prediction 
using different architectures. However, the decision for using a Transformer network is argued as follows:
- Transformer achieved in 13/15 ASR benchmarks a better performance than RNN [1]
- Transformer are Turing-complete and can therefore model almost any sequence models [2]
- Transformer have lower training-cost than RNN due to their self-attention [3]
- Transformer can easily be combined with encoder and decoder architectures


Suitable values must be found for the hyperparameters $$n$$ and $$k$$. At the beginning, it is recommended to start with a 
simple task. This means that $$n$$ should be large and $$k$$ small. After that, one of the two parameters should be fixed 
while the other is changed linearly. Determining the initial as well as final parameters $$n$$ and $$k$$ is quite difficult.

The task is closely related to next-word prediction in NLP systems. When more data is given, more accurate predictions 
can generally be made. Therefore, at least one or two complete sentence should be given at the beginning. One sentence consists
on average of 20 words [4] (but this number is highly dependend on the dataset, older text consisted of avg. 60 words).
The average speaker says 120-200 words per minute [5].

To start with a rather simple example, we assume that 2 sentences must be given. To start with a rather simple example, 
we assume that $$2$$ sentences must be given. With an average sentence length of $$20$$ words and an average speaker saying $$160$$
words per minute, this corresponds to:

$$
$n$ = 40/160 = 0.25 \text{min.} = 15 \text{sec.}
$$

At the beginning only 1 frame should be predicted with it. Most NLP systems predict only a few words reliably. The number of correctly predicted words is strongly dependent on the data set and the input length. It is assumed that no more than 1-2 words can be predicted. 1 word corresponds in average to 

$$
t_{\text{1 word}} = 60 / 160 = 0.375s
$$

, two words corresponds to

$$
t_{\text{2 words}} = 2 \cdot 60 / 160 = 0.75s
$$

Therefore, the initial parameters are defined as follows:
- $$n = 15 \text{sec.} = 1200 \text{frames} $$
- $$k = 1 \text{frames} $$
  
Then $$k$$ is continously increased until it has the size $$k = 0.375 \text{sec.} = 30 \text{frames} $$ (= predict one word).
If this still works, $$k$$ can be increased to two words, meaning $$k = 0.75 \text{sec.} = 60 \text{frames} $$


[1] S. Karita et al., "A Comparative Study on Transformer vs RNN in Speech Applications," 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)

[2] Pérez et al., "On the Turing Completeness of Modern Neural Network Architectures," 2019 ICLR 

[3] Vaswani et al, "Attention Is All you Need", 2017 NIPS

[4] Moore, A. (December 22, 2011). The long sentence: A disservice to science in the Internet age. Retrieved November 23, 2015, from Wiley website: http://exchanges.wiley.com/blog/2011/12/22/the-long-sentence-a-disservice-to-science-in-the-internet-age/

[5] Huang et al "Speech Rate and Pausing in English: Comparing learners at different levels of proficiency with native speakers"

# Current Status

The application is still under development. So far, only a few experiments were conducted:


## Predict a Segment in a MFCC
1. Different waveforms from
   the [SPEECHCOMMANDS](https://pytorch.org/audio/stable/_modules/torchaudio/datasets/speechcommands.html) dataset are
   loaded. One command in the dataset is a men who says "eight":
   <p align="center">
      <img src="assets/concept/waveform.png" alt="Waveform of the signal" width="50%" />
      <br>
      <audio controls>
         <source src="assets/concept/waveform.wav" type="audio/wav">
      </audio>
   </p>


2. From this signal, the MFCC is calculated:
   <p align="center">
      <img src="assets/concept/mfcc.png" alt="MFCC of the Waveform" width="50%" />
      <br>
      <audio controls>
         <source src="assets/concept/MFCC.wav" type="audio/wav">
      </audio>
   </p>


2. Then a part of the signal is masked (e.g. an area in the center is set to 0):
   <p align="center">
      <img src="assets/concept/mfcc_masked.png" alt="MFCC of the Waveform" width="50%" />
      <br>
      <audio controls>
         <source src="assets/concept/MFCC_masked.wav" type="audio/wav">
      </audio>
   </p>

3. This masked signal is fed into a transformer network which was trained to predict the masked part. If the (masked)
   input signal is composed with the prediction, the signal can be reconstructed (the masked part of the signal is
   replaced with the prediction from the transformer network):
   <p align="center">
      <img src="assets/concept/mfcc_reconstructed.png" alt="MFCC of the Waveform" width="50%" />
      <br>
      <i>On the left the ground truth mask, on the right the predicted mask from the network</i>
      <br>
      <audio controls>
         <source src="assets/concept/MFCC_reconstructed.wav" type="audio/wav">
      </audio>
      <br>
      <br>
      <i>This is the MFCC signal, predicted by the network.</i>
   </p>
   
## Predict Different Number of Frames
At the end of an MFCC, $$k$$ frames were omitted. Then, a transformer network was trained to predict the missing $$k$$ frames.

Different Input-Data:

| $$k=1$$ | $$k=5$$ | $$k=10$$ |
|---------|---------|----------|
| <img src="assets/concept/input_1f_masked.png"/> | <img src="assets/concept/input_5f_masked.png" /> | <img src="assets/concept/input_10f_masked.png" /> |

<p align="center">
 ...
</p>

| $$k=50$$ | $$k=60$$ | $$k=70$$ |
|----------|----------|----------|
| <img src="assets/concept/input_50f_masked.png" /> | <img src="assets/concept/input_60f_masked.png" /> | <img src="assets/concept/input_70f_masked.png"/> |


The error tends to be slightly larger when more frames are masked (larger $$k$$). However, this is not true in all cases and the difference is relatively small.

###### Error per Epoch for Runs with different k

| MAE | MSE |
|-----|-----|
| <img src="assets/concept/mae_ex2.png" /> | <img src="assets/concept/mse_ex2.png" /> | 


###### Standard Deviation per Epoch for Runs with different k

| MAE | MSE |
|-----|-----|
| <img src="assets/concept/mae_std_ex2.png" /> | <img src="assets/concept/mse_std_ex2.png" /> | 



