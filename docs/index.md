---
layout: default
title: Index
nav_exclude: true
---

# Importance of the Temporal Context for ASR
This is the documentation of the repository [https://github.com/sagerpascal/temporal-speech-context](https://github.com/sagerpascal/temporal-speech-context).
It was created by [Pascal Sager](https://sagerpascal.github.io) as his first project thesis. The project thesis is
supervised by [Prof. Dr. Thilo Stadelmann](https://stdm.github.io), who is an expert in Deep Learning and very experienced in ASR.


## Description, Problem Definition
In recent years, there has been significant progress in the field of speech- and speaker-recognition. Thereby, various features 
are extracted from the audio signals and then analyzed. Stadelmann [1] showed the importance of the temporal context 
within audio signals. So far, it has been assumed that CNN's capture this temporal context through their receptive 
field. However, Neururer [2] showed, that CNN's achieve about the same recognition accuracy even it the temporal context 
is destroyed by mixing the individual frames. This implies that the temporal context is still contained in the audio signals, 
but not exploited with today's state-of-the-art architectures.

In this project thesis, the temporal context will be investigated. For this purpose, an audio signal is transformed into 
MFCC frames. Then, a neural network is trained to predict a preceding or following frames for a given MFCC.
The training may be inspired by modern NLP algorithms, in which a part of the data is masked and then predicted based on 
it's surrounding. For the evaluation of the predicted frames, appropriate distance metrics must be evaluated. In 
addition, perception experiments will be conducted in collaboration with the University of Zurich.

The results from this project can show the presence of the temporal context within speech signals. Depending on how many 
frames can be predicted, the importance of temporal features could be inferred. Based on the results, 
it can be concluded whether these temporal features should be considered by future ASR research.
During the project, the student should perform the following tasks:

- perform a literature study on the current state-of-the-art
- select a baseline approach and adapt it to the given problem space
- setting up the development environment and train thr neural network
- Improve and evaluate the network iteratively
- Assist UZH with perception experiments as necessary
- Publish a scientific paper in collaboration with ZHAW and UZH (if the circumstances allow)
- Write a scientific report with focus on motivation, methods, argumentation and results (about 10 pages with sources)


[1] Stadelmann, "Voice Modeling Methods for Automatic Speaker Recognition", [http://archiv.ub.uni-marburg.de/diss/z2010/0465](http://archiv.ub.uni-marburg.de/diss/z2010/0465)

[2] Neururer, ????

