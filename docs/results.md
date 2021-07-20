---
layout: default
title: Results
nav_order: 1
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

[comment]: <> (### Simple Baseline MFCC)

[comment]: <> (The goal is to predict the $$k$$ following frames for $$n$$ given frames. In order to estimate in which range an MSE from a working network should lie, two simple baselines are defined:)

[comment]: <> (- *Mean of previous $$n$$ frames:* The average value on the time axis of the $$k$$ given frames is calculated. This mean value is then used as prediction for all of the following $$k$$ values. )

[comment]: <> (- *Last given frame:* The last given frame is used as prediction for all of the following $$n$$ values. This prediction is fairly good for the first frame, but quickly becomes inaccurate for predictions further away)

[comment]: <> (<p align="center">)

[comment]: <> (      <img src="assets/results/baseline_predictions_s=150.png" alt="Results Baseline Predictions" width="70%" />)

[comment]: <> (      <br>)

[comment]: <> (      <i> Figure 2: The average value as prediction has a constant MSE of about 0.14. The last value, on the other hand, achieves a small MSE for the first few frames, but the MSE increases rapidly thereafter.</i>)

[comment]: <> (</p>)

[comment]: <> (A well-working neural network should be able to achieve better results than these baselines. This means that the MSE of the network should be smaller and in the lower right quadrant of the graph shown in figure 2.)

[comment]: <> (### Baseline Network for Mel-Spectrogram)

[comment]: <> (This repository mainly investigates the prediction of MFCC frames. However, in order to create a simple baseline with computer vision methods, Mel-Spectrograms were also used. The features of Mel-Spectrograms are highly dependent and therefore easier to predict with computer vision methods. Thereby, the principle is identical to the prediction of MFCCs, as shown in Figure 3.)

[comment]: <> (<p align="center">)

[comment]: <> (      <img src="assets/results/mel-spectrogram-example.png" alt="Example Mel-Spectrogram" width="70%" />)

[comment]: <> (      <br>)

[comment]: <> (      <i> Figure 3: A given Mel-Spectrogram &#40;upper plot&#41; is split into two sub-spectrograms: The first n frames are used as input data x &#40;lower left&#41; and the following k frames have to be predicted by the network &#40;lower right&#41;.</i>)

[comment]: <> (</p>)

[comment]: <> (The same baselines were calculated for the Mel-Spectrogram as done for the MFCC: The average value of the previous $$n$$ frames and the last given frame were used as prediction for the following $$k$$ frames. A neural network should be able to outperform these simple baseline approaches. )

[comment]: <> (As a first approach, a U-Net++ was implemented. The U-Net++ is very well proven and usually achieves good results. The network used is a slight modification compared to the original version: An Efficientnet-b0 was used as the encoder. In total, the encoder and the decoder have a depth of $$d=5$$ and at the end, a linear layer maps the calculated prediction to the correct length.)

[comment]: <> (To make pooling simpler to implement, powers of two were used as the input length $$n$$. Since the segment length $$s_{max}$$ should be smaller than $$150$$ &#40;i.e. $$s_{max} \leq 150$$&#41;, $$n=64$$ frames were used as input. Thus, more examples are available for training, but less context is available &#40;shorter segments given&#41;.)

[comment]: <> (Figure 4 shows that this U-Net++ implementation was able to outperform the two simple baselines.)

[comment]: <> (<p align="center">)

[comment]: <> (      <img src="assets/results/baseline_predictions_mel-spectro_s=32.png" alt="Baseline Mel-Spectrogram" width="70%" />)

[comment]: <> (      <br>)

[comment]: <> (      <i> Figure 4: The prediction of the U-Net++ was better than the two simple baselines. However, the plot also shows that the network is overfitting on the training data.</i>)

[comment]: <> (</p>)

[comment]: <> (If only one frame has to be predicted &#40;$$k=1$$&#41;, then using the last given frame is considered a good prediction. This prediction has an average MSE of $$0.15$$. Therefore, predictions with a MSE $$\leq 0.15$$ are in general considered as good predictions. )

[comment]: <> (However, the U-Net++ achieves such results only for very short sequences. Moreover, the results are not satisfactory when viewed visually or acoustically, despite the small MSE.)

[comment]: <> (Figure 5 shows the predicted Mel-Spectrogram of the U-Net on the training set. The MSE was xxxxxxxxxx and therefore below the threshold of $$0.15$$.)

[comment]: <> (The plot indicates that this prediction was not good despite the low MSE. Instead of the network predicting phenomes, it learns to create a "noise" with a low MSE. This suggests that the MSE is not necessarily a suitable loss function. This is examined in more detail in the following section.)

[comment]: <> (> TODO: add plot / sound -> U-Net snowy-snow-17 on Training set)

[comment]: <> (<p align="center">)

[comment]: <> (      <img src="assets/results/todo_add_plot.png" alt="Predicted Mel-Spectrgramm" width="70%" />)

[comment]: <> (      <br>)

[comment]: <> (      <i> Figure 5: Prediction of 8 frames of a Mel-Spectrogram, created with a U-Net++ on the training set</i>)

[comment]: <> (</p>)

## Predictions
The audio files from the TIMIT dataset were converted to Mel-spectrograms. Fixed length segments were then extracted from these Mel-spectrograms using a sliding window. The segments were split into two parts: The first part was fed into the model and used to predict the second part of the segment.
This process is illustrated in below figure 2.

<p align="center">
      <img src="assets/results/concept.PNG" alt="Concept" width="70%" />
      <br>
      <i> Figure 2: Concept of the trained network</i>
</p>

Some results are presented below. In these models, 120 frames were fed into the network and 25 frames were predicted.
After that, the sliding window was shifted forward by 1 frame and the process was repeated.
The 25 predicted frames correspond approximately to one word. However, the quality of the prediction is difficult to determine based on only one word. Therefore, several predictions were composed. 

The composition was done as follows: From a predicted sequence, the frame at the position `offset` was stored. Then the sliding window was moved shifted by 1 and the next frame at the position `offset` was saved.

The prediction becomes of course more difficult if the `offset` is larger. For example, with an `offset=25`, 24 frames must first be predicted and then only the 25th predicted frame is stored. With a smaller `offset` less frames have to be predicted and thus the task becomes easier.

The results are presented below. Only the ground truth and the prediction are shown in the table -> the segment that was fed into the network is not visible.

#### Speaker MRJM4, Sentence SA1

Ground truth audio (resynthesized Mel-spectrogram):

<p align="center">
      <audio controls>
         <source src="assets/results/MRJM4-SA1/original.wav" type="audio/wav">
      </audio>
</p>


| `offset` | Ground Truth spectrogram | Predicted Spectrogram | Predicted Audio |
|----------|--------------------------|-----------------------|-----------------|
| 1        | <img src="assets/results/MRJM4-SA1/1-ahead_gt.png" width="75%" /> | <img src="assets/results/MRJM4-SA1/1-ahead.png" width="75%" /> | <audio controls><source src="assets/results/MRJM4-SA1/1-ahead.wav" type="audio/wav"></audio> |
| 5        | <img src="assets/results/MRJM4-SA1/5-ahead_gt.png" width="75%" /> | <img src="assets/results/MRJM4-SA1/5-ahead.png" width="75%" /> | <audio controls><source src="assets/results/MRJM4-SA1/5-ahead.wav" type="audio/wav"></audio> |
| 10       | <img src="assets/results/MRJM4-SA1/10-ahead_gt.png" width="75%" /> | <img src="assets/results/MRJM4-SA1/10-ahead.png" width="75%" /> | <audio controls><source src="assets/results/MRJM4-SA1/10-ahead.wav" type="audio/wav"></audio> |
| 20       | <img src="assets/results/MRJM4-SA1/20-ahead_gt.png" width="75%" /> | <img src="assets/results/MRJM4-SA1/20-ahead.png" width="75%" /> | <audio controls><source src="assets/results/MRJM4-SA1/20-ahead.wav" type="audio/wav"></audio> |
| 25       | <img src="assets/results/MRJM4-SA1/25-ahead_gt.png" width="75%" /> | <img src="assets/results/MRJM4-SA1/25-ahead.png" width="75%" /> | <audio controls><source src="assets/results/MRJM4-SA1/25-ahead.wav" type="audio/wav"></audio> |


#### Speaker MJTH0, Sentence SA1 (Overfitted Model)

*This model overfitted on the test data - but is for some sentences (such as this one) particularly good*

Ground truth audio (resynthesized Mel-spectrogram):

<p align="center">
      <audio controls>
         <source src="assets/results/MJTH0-SA1(overfitted)/original.wav" type="audio/wav">
      </audio>
</p>


| `offset` | Ground Truth spectrogram | Predicted Spectrogram | Predicted Audio |
|----------|--------------------------|-----------------------|-----------------|
| 1        | <img src="assets/results/MJTH0-SA1(overfitted)/1-ahead_gt.png" width="75%" /> | <img src="assets/results/MJTH0-SA1(overfitted)/1-ahead.png" width="75%" /> | <audio controls><source src="assets/results/MJTH0-SA1(overfitted)/1-ahead.wav" type="audio/wav"></audio> |
| 5        | <img src="assets/results/MJTH0-SA1(overfitted)/5-ahead_gt.png" width="75%" /> | <img src="assets/results/MJTH0-SA1(overfitted)/5-ahead.png" width="75%" /> | <audio controls><source src="assets/results/MJTH0-SA1(overfitted)/5-ahead.wav" type="audio/wav"></audio> |
| 10       | <img src="assets/results/MJTH0-SA1(overfitted)/10-ahead_gt.png" width="75%" /> | <img src="assets/results/MJTH0-SA1(overfitted)/10-ahead.png" width="75%" /> | <audio controls><source src="assets/results/MJTH0-SA1(overfitted)/10-ahead.wav" type="audio/wav"></audio> |
| 20       | <img src="assets/results/MJTH0-SA1(overfitted)/20-ahead_gt.png" width="75%" /> | <img src="assets/results/MJTH0-SA1(overfitted)/20-ahead.png" width="75%" /> | <audio controls><source src="assets/results/MJTH0-SA1(overfitted)/20-ahead.wav" type="audio/wav"></audio> |
| 25       | <img src="assets/results/MJTH0-SA1(overfitted)/25-ahead_gt.png" width="75%" /> | <img src="assets/results/MJTH0-SA1(overfitted)/25-ahead.png" width="75%" /> | <audio controls><source src="assets/results/MJTH0-SA1(overfitted)/25-ahead.wav" type="audio/wav"></audio> |




#### Speaker MCHH0, Sentence SA2

Ground truth audio (resynthesized Mel-spectrogram):

<p align="center">
      <audio controls>
         <source src="assets/results/MCHH0-SA2/original.wav" type="audio/wav">
      </audio>
</p>


| `offset` | Ground Truth spectrogram | Predicted Spectrogram | Predicted Audio |
|----------|--------------------------|-----------------------|-----------------|
| 1        | <img src="assets/results/MCHH0-SA2/1-ahead_gt.png" width="75%" /> | <img src="assets/results/MCHH0-SA2/1-ahead.png" width="75%" /> | <audio controls><source src="assets/results/MCHH0-SA2/1-ahead.wav" type="audio/wav"></audio> |
| 5        | <img src="assets/results/MCHH0-SA2/5-ahead_gt.png" width="75%" /> | <img src="assets/results/MCHH0-SA2/5-ahead.png" width="75%" /> | <audio controls><source src="assets/results/MCHH0-SA2/5-ahead.wav" type="audio/wav"></audio> |
| 10       | <img src="assets/results/MCHH0-SA2/10-ahead_gt.png" width="75%" /> | <img src="assets/results/MCHH0-SA2/10-ahead.png" width="75%" /> | <audio controls><source src="assets/results/MCHH0-SA2/10-ahead.wav" type="audio/wav"></audio> |
| 20       | <img src="assets/results/MCHH0-SA2/20-ahead_gt.png" width="75%" /> | <img src="assets/results/MCHH0-SA2/20-ahead.png" width="75%" /> | <audio controls><source src="assets/results/MCHH0-SA2/20-ahead.wav" type="audio/wav"></audio> |
| 25       | <img src="assets/results/MCHH0-SA2/25-ahead_gt.png" width="75%" /> | <img src="assets/results/MCHH0-SA2/25-ahead.png" width="75%" /> | <audio controls><source src="assets/results/MCHH0-SA2/25-ahead.wav" type="audio/wav"></audio> |


#### Speaker FJLM0, Sentence SA2 (Overfitted Model)

*This model overfitted on the test data - but is for some sentences (such as this one) particularly good*

Ground truth audio (resynthesized Mel-spectrogram):

<p align="center">
      <audio controls>
         <source src="assets/results/FJLM0-SA2(overfitted)/original.wav" type="audio/wav">
      </audio>
</p>


| `offset` | Ground Truth spectrogram | Predicted Spectrogram | Predicted Audio |
|----------|--------------------------|-----------------------|-----------------|
| 1        | <img src="assets/results/FJLM0-SA2(overfitted)/1-ahead_gt.png" width="75%" /> | <img src="assets/results/FJLM0-SA2(overfitted)/1-ahead.png" width="75%" /> | <audio controls><source src="assets/results/FJLM0-SA2(overfitted)/1-ahead.wav" type="audio/wav"></audio> |
| 5        | <img src="assets/results/FJLM0-SA2(overfitted)/5-ahead_gt.png" width="75%" /> | <img src="assets/results/FJLM0-SA2(overfitted)/5-ahead.png" width="75%" /> | <audio controls><source src="assets/results/FJLM0-SA2(overfitted)/5-ahead.wav" type="audio/wav"></audio> |
| 10       | <img src="assets/results/FJLM0-SA2(overfitted)/10-ahead_gt.png" width="75%" /> | <img src="assets/results/FJLM0-SA2(overfitted)/10-ahead.png" width="75%" /> | <audio controls><source src="assets/results/FJLM0-SA2(overfitted)/10-ahead.wav" type="audio/wav"></audio> |
| 20       | <img src="assets/results/FJLM0-SA2(overfitted)/20-ahead_gt.png" width="75%" /> | <img src="assets/results/FJLM0-SA2(overfitted)/20-ahead.png" width="75%" /> | <audio controls><source src="assets/results/FJLM0-SA2(overfitted)/20-ahead.wav" type="audio/wav"></audio> |
| 25       | <img src="assets/results/FJLM0-SA2(overfitted)/25-ahead_gt.png" width="75%" /> | <img src="assets/results/FJLM0-SA2(overfitted)/25-ahead.png" width="75%" /> | <audio controls><source src="assets/results/FJLM0-SA2(overfitted)/25-ahead.wav" type="audio/wav"></audio> |

