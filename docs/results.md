---
layout: default
title: Results
nav_order: 4
---

# Results

## TIMIT

### Parameters $$n$$ & $$k$$
The choice of the parameters $$n$$ and $$k$$ is very important. As Ref. [1] described, larger $$n$$ often lead to better results. 
However, the TIMIT data set contains rather short audio sequences ($$\approx 3s $$) and therefore the parameter $$nn$$ cannot be arbitrarily large. 
Figure 1 shows the different number of MFCC frames per audio file:

<p align="center">
      <img src="assets/results/timit-mfcc-lengths.png" alt="MFCC Lengths" width="70%" />
      <br>
      <i> Figure 1: Number of MFCC frames per file</i>
</p>

In order to have enough data for training, $$n + k$$ should not be longer than $$\approx 150$$ frames.
Thus, the following constraint applies:

$$
s = n + k_{max} \leq 150
$$

The chapter [concept]({{ site.baseurl }}{% link index.md %}) describes, that one word corresponds to $$30$$ frames.
This means that $$n + k$$ frames together consists of approx. 5 words.


[1] Dumbali and Nagaraja, "Real Time Word Prediction Using N-Grams Model", 2019, International Journal of Innovative Technology and Exploring Engineering (IJITEE)

### Simple Baseline MFCC
The goal is to predict the $$k$$ following frames for $$n$$ given frames. In order to estimate in which range an MSE from a working network should lie, two simple baselines are defined:

- *Mean of previous $$n$$ frames:* The average value on the time axis of the $$k$$ given frames is calculated. This mean value is then used as prediction for all of the following $$k$$ values. 
- *Last given frame:* The last given frame is used as prediction for all of the following $$n$$ values. This prediction is fairly good for the first frame, but quickly becomes inaccurate for predictions further away

<p align="center">
      <img src="assets/results/baseline_predictions_s=150.png" alt="Results Baseline Predictions" width="70%" />
      <br>
      <i> Figure 2: The average value as prediction has a constant MSE of about 0.14. The last value, on the other hand, achieves a small MSE for the first few frames, but the MSE increases rapidly thereafter.</i>
</p>

A well-working neural network should be able to achieve better results than these baselines. This means that the MSE of the network should be smaller and in the lower right quadrant of the graph shown in figure 2.

### Baseline Network for Mel-Spectrogram
This repository mainly investigates the prediction of MFCC frames. However, in order to create a simple baseline with computer vision methods, Mel-Spectrograms were also used. The features of Mel-Spectrograms are highly dependent and therefore easier to predict with computer vision methods. Thereby, the principle is identical to the prediction of MFCCs, as shown in Figure 3.

<p align="center">
      <img src="assets/results/mel-spectrogram-example.png" alt="Example Mel-Spectrogram" width="70%" />
      <br>
      <i> Figure 3: A given Mel-Spectrogram (upper plot) is split into two sub-spectrograms: The first n frames are used as input data x (lower left) and the following k frames have to be predicted by the network (lower right).</i>
</p>

The same baselines were calculated for the Mel-Spectrogram as done for the MFCC: The average value of the previous $$n$$ frames and the last given frame were used as prediction for the following $$k$$ frames. A neural network should be able to outperform these simple baseline approaches. 

As a first approach, a U-Net++ was implemented. The U-Net++ is very well proven and usually achieves good results. The network used is a slight modification compared to the original version: An Efficientnet-b0 was used as the encoder. In total, the encoder and the decoder have a depth of $$d=5$$ and at the end, a linear layer maps the calculated prediction to the correct length.

To make pooling simpler to implement, powers of two were used as the input length $$n$$. Since the segment length $$s_{max}$$ should be smaller than $$150$$ (i.e. $$s_{max} \leq 150$$), $$n=64$$ frames were used as input. Thus, more examples are available for training, but less context is available (shorter segments given).
Figure 4 shows that this U-Net++ implementation was able to outperform the two simple baselines.

<p align="center">
      <img src="assets/results/baseline_predictions_mel-spectro_s=32.png" alt="Baseline Mel-Spectrogram" width="70%" />
      <br>
      <i> Figure 4: The prediction of the U-Net++ was better than the two simple baselines. However, the plot also shows that the network is overfitting on the training data.</i>
</p>

If only one frame has to be predicted ($$k=1$$), then using the last given frame is considered a good prediction. This prediction has an average MSE of 0.15. Therefore, predictions with a MSE $$\leq 0.15$$ are in general considered as good predictions. 






#### Why it is Challenging
The TIMIT data set consists of different speakers saying the same sentences. Predicting the following words based on some given words would be easy if it would be text. This is because each word can be represented by a token. For audio files, however, this is much more complex. A single word consists of several frames. In addition, each speaker pronounces the words differently (speed, pitch, glitches, pauses, mood, ...). Thus, the words cannot be represented by a token but by a 2D vector (features x time). 

This issue is demonstrated in the following. A typical phrase from the TIMIT data set is "She had your dark suit in greasy wash water all year". The entire data set was searched for two speakers who pronounce this sentence as identically as possible (pitch and timing). In addition, the files were trimmed so that they are as identical as possible. Figure 5 shows the Mel-Spectrogram of these two audio files.

<p align="center">
      <img src="assets/results/speaker_comparison_male.png" alt="Mel-Spectrograms" width="70%" />
      <br>
      <i> Figure 5: The Mel-Spectrograms of two men saying the sentence "She had your dark suit in greasy wash water all year" as similar as possible.</i>
</p>

The MSE between these two spectrograms is $$MSE=0.3917$$, which is a good result. However, with this result, there is also some luck involved. The whole procedure was repeated. Figure 6 shows two more Mel-Spectrograms, which match well optically. 

<p align="center">
      <img src="assets/results/speaker_comparison_female.png" alt="Mel-Spectrograms" width="70%" />
      <br>
      <i> Figure 6: The Mel-Spectrograms of two women saying the sentence "She had your dark suit in greasy wash water all year" as similar as possible.</i>
</p>

These two spectrograms have an MSE of 2.981. This is worse than simply taking the average of the previous frames as the prediction. 

