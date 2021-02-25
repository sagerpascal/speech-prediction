---
layout: default
title: Index
nav_exclude: true
---

# Importance of the Temporal Context for ASR
This is the documentation of the repository [https://github.com/sagerpascal/temporal-speech-context](https://github.com/sagerpascal/temporal-speech-context).
It was created by [Pascal Sager](https://sagerpascal.github.io) as his first project thesis. Goal of the thesis:

> "Train a NN to predict a frame of speech (MFCC) from its predecessor; Could be done with a transformer trained in the
> same way current NLP models are trained."

## Description, Problem Definition
In recent years, there has been significant progress in the field of speech- and speaker-recognition. Thereby, various features 
are extracted from the audio signals and then analyzed. Stadelmann [1] has shown that the temporal context is an important 
feature within audio signals. So far, it has been assumed that CNN's capture this temporal context through their receptive 
field. However, Neururer [2] showed, that CNN's achieve about the same recognition accuracy even it the temporal context 
is destroyed by mixing the individual frames. This implies that the temporal context is still contained in the audio signals, 
but not exploited with today's state-of-the-art architectures.

In this Project Thesis, the temporal context within MFCC's will be investigated. For this purpose, an audio signal is 
transformed into a MFCC. Then, a neural network is trained to predict a preceding or following frame based on a given MFCC.
The training may be inspired by modern NLP algorithms such as n-gram models, in which part of the data is masked 
and then predicted. For the evaluation of the predicted frames, appropriate distance metrics must be evaluated. In 
addition, perception experiments will be conducted in collaboration with the University of Zurich.

The results from this project show the importance of temporal context within speech signals. Based on the results, 
it can be concluded whether these temporal features are important and should be considered by future ASR research.
During the project,  the student should perform the following tasks:

- perform a literature study on the current state-of-the-art
- select a baseline approach and adapt it to the given problem space
- setting up the development environment and train thr neural network
- Improve and evaluate the network iteratively
- Assist UZH with perception experiments as necessary
- Publish a scientific paper in collaboration with ZHAW and UZH (if the circumstances allow)
- Write a scientific report with focus on motivation, methods, argumentation and results (about 10 pages with sources)


[1] Stadelmann, "Voice Modeling Methods for Automatic Speaker Recognition", [http://archiv.ub.uni-marburg.de/diss/z2010/0465](http://archiv.ub.uni-marburg.de/diss/z2010/0465)

[2] Neururer, ????

