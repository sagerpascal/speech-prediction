---
layout: default
title: Concept
nav_order: 1
---

# Concept

TODO

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
      <i>On the left the groud truth mask, on the right the predicted mask from the network</i>
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



