---
layout: default
title: Concept
nav_order: 1
---

# Concept

TODO

# Current Status

The application is still under development. So far, only a proof of concept has been created:

1. Different waveforms from
   the [SPEECHCOMMANDS](https://pytorch.org/audio/stable/_modules/torchaudio/datasets/speechcommands.html) dataset are
   loaded. One command in the dataset is a woman who says "tree":
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
   


