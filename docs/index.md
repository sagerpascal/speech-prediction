---
layout: default
title: Index
nav_exclude: true
---

# Modeling Temporal Features of Speech with Sequence Learners for Frame Predictions
This is the documentation of the repository [https://github.com/sagerpascal/temporal-speech-context](https://github.com/sagerpascal/temporal-speech-context).
It was created by [Pascal Sager](https://sagerpascal.github.io) as his first project thesis. The project thesis is
supervised by [Prof. Dr. Thilo Stadelmann](https://stdm.github.io), who is an expert in Deep Learning and very experienced in speaker recognition.


## Description, Problem Definition
Recently there has been significant progress in the field of speaker recognition thanks to advances in deep learning. Stadelmann [1] previously showed the importance of the temporal context within the audio signal as a feature. So far [2], it has been assumed that deep learning methods like CNNs and RNNs capture these temporal aspects through their receptive field when operating on spectrograms. However, Neururer [3] showed that they achieve about the same recognition accuracy even when the temporal context is destroyed by shuffling the individual audio frames. This implies that state-of-the-art deep learning architectures do not exploit temporal information properly and potentially do not even model it.

In this project, modeling the temporal context explicitly by deep sequence learning architectures will be investigated. For this purpose, an audio signal is transformed into a sequence of MFCC frames. Then, a neural network is trained to predict preceding or next frame(s) for a given context of MFCCs, inspired by modern NLP methods that are trained on parts of the text to predict surrounding words. As in NLP, the goal of this form of training is to use these models that explicitly learned about temporal context for subsequent tasks like e.g. speaker recognition. The evaluation of the outcomes will rely partially on the performance on such tasks, and on perception experiments to be conducted in collaboration with the University of Zurich (Volker Dellwo's group). Specifically:

- Perform literature research on the current state-of-the-art in modeling temporal features for automatic speaker recognition, especially through context prediction

- Select baseline approach and adapt it to the given problem space

- Set up development environment and train (improve & evaluate) neural networks iteratively

- Assist UZH staff with perception experiments as necessary

- Publish a scientific paper with ZHAW and UZH researchers (if circumstances allow)

- Write a scientific report with focus on motivation, methods, argumentation, and results (paper style, about 10 pages)

[1] Stadelmann and Freisleben. "Unfolding Speaker Clustering Potential: A Biomimetic Approach". ACM MM'09.

[2] Stadelmann et al. "Capturing Suprasegmental Features of a Voice with RNNs for Improved Speaker Clustering".  ANNPR'18.

[3] Neururer. "Exploiting the Full Information of Varying-Length Utterances for DNN-Based Speaker Verification". Master Thesis, ZHAW, 2020.


