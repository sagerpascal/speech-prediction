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

### Simple Baseline
The goal is to predict the $$k$$ following frames for $$n$$ given frames. In order to estimate in which range an MSE from a working network should lie, two simple baselines are defined:

- *Mean of previous $$k$$ frames:* The average value on the time axis of the $$k$$ given frames is calculated. This mean value is then used as prediction for all of the following $$n$$ values. 
- *Last given frame:* The last given frame is used as prediction for all of the following $$n$$ values. This prediction is fairly good for the first frame, but quickly becomes inaccurate for predictions further away

<p align="center">
      <img src="assets/results/baseline_predictions_s=150.png" alt="Results Baseline Predictions" width="70%" />
      <br>
      <i> Figure 2: The average value as prediction has a constant MSE of about 0.14. The last value, on the other hand, achieves a small MSE for the first few frames, but the MSE increases rapidly thereafter.</i>
</p>

A well-working neural network should be able to achieve better results than these baselines. This means that the MSE of the network should be smaller and in the lower right quadrant of the graph shown in figure 2.