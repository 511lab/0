# STTSC
#A parameter-free self-training algorithm based on the three successive confirmation rule

We provide the codes and the datasets.

Abstract：Semi-supervised learning is a popular research topic today, and self-training is a classical semi-supervised learning framework. How to select high-confidence samples in self-training is a very important step. However, the existing algorithms do not consider both global and local information of the data. In the paper, we propose a parameter-free self-training algorithm based on the three successive confirmation rule, which combines global and local information to identify high-confidence samples. Concretely, the local information of the samples is obtained by using k nearest neighbor and the global information of the samples is derived from the three successive confirmation rule, this dual selection strategy helps to improve the quality of the high-confidence samples and further improve the performance of classification. We conducted experiments on 14 benchmark datasets and compared our method with other self-training algorithms. We use accuracy and F-score as performance metrics. The experimental results demonstrate that our algorithm achieves significant performance improvement in classification tasks, proving its effectiveness and superiority in semi-supervised learning. All the codes and data sets can be found on website https://github.com/511lab/STTSC.

# Set up
## Requirements
All the experiments were conducted with 64G RAM, 64-bit Windows 10 and Inter Core i9 processor. 
All the codes are implemented with MATLAB 2023a. 

In order to keep consistent with the comparison algorithms, the 3NN classification is chosen as the base classifier.  The operating parameters of the 3NN classification are as follows: Euclidean distance is used to calculate the intersample distance, and the number of sample nearest neighbors is taken as 3.  

# Datasets
You can also download the datasets from
link: https://github.com/511lab/STTSC/datasets

# Codes 
Source Codes on Matlab are available at,   
https://github.com/511lab/STTSC/code

1.main.m # The main function of Classification performance.

2.Ntest.m # The main function of Noise experiment.

3.STTSC.m # The function of our proposed STTSCalgorithm.

4.Random_sampling.m # The function of the Stratified sampling of the Classification performance.

6.noise_sampling.m # The function of the Sampling ratio of the Noise experiment.

(1) Classification performance

 Download the codes on https://github.com/511lab/STTSC/code, then run the main.m code.
 
(2) Noise experiment

Download the codes on https://github.com/511lab/STTSC/code, then run the Ntest.m code.

# Contact: 
For any problem about this dataset or codes, please contact Dr. Wang (wjkweb@163.com).


