# CNNwithAttention

This repository contains the code for experimenting with the use of Attention Mechanism in Convolutional Neural Networks (CNNs). Specifically, we test different different models of CNNs on the Fashion MNIST dataset. The details of each scripts are given below.

## Scripts in this repository

### common.py

Common code for creating the loader for training/test data, and evaluating acccuracy of the model on the test data. This ensures that we have a uniform training/testing set-up across all the different models that we are exploring.


### fashion_mnist.py

Standard CNN implementation for image classification using Convolutional and Max Pool layers for image classification.


### fashion_mnist_attn.py

CNN implementation in which the Max Pool layer is replaced by a Single-Head Attention Mechanism.


### fashion_mnist_multi_attn.py

CNN implementation in which the Max Pool layer is replaced by a Multi-Headed Attention Mechanism.


## Hardware configuration

The specific configuration of the machine on which I am running these scripts is:

1. CPU: AMD Ryzen 9 7900X 12-Core Processor, 24 vCPU
2. GPU: RTX 4080 Super
3. Memory: 64 GB RAM


## Model performance

As of Mar 24, 2026, here is the current performance of the models on the Fashion MNIST dataset:

1. The basic CNN model in `fashion_mnist.py` script has accuracy generally in the range of 92.4% - 92.6%.
2. The CNN model with multi-head attention in `fashion_mnist_multi_attn.py` script achieves accuracy in the range of 93.3% - 93.5%.
3. The CNN model with single-head attention in `fashion_mnist_attn.py` script accuracy falls in between these two ranges.

The goal is to continue refining the multi-head attention model to improve its accuracy beyond the 95% range.

## Model updates

### Mar 25, 2026

Changes incorporated:

1. Included a Global Transformer Block for feature extraction as the last layer of CNN, in addition to the multi-head attention layers
2. Using only a two-layer feed-forward network for the classification layer
3. Incorporated `label_smoothing` with value 0.1 for the cross-entropy loss function in model training
4. Increased the number of epochs to 60

With these changes, the resulting accuracy on test dataset is now 93.9%, the highest we have been able to achieve so far.