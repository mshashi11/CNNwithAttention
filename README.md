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

### Mar 27, 2026 (Exp 31)

Changes incorporated:
1. Added an additional 3x3 `Conv2d` + `BatchNorm2d` + `ReLU` layer in the first feature block to increase feature extraction depth.
2. Switched from `GroupNorm` to `BatchNorm2d` in `MultiHeadAttentionPool2d` for better normalization stability.
3. Maintained successful features from Exp 27: learned `pos_embed` in `GlobalTransformerBlock`, channels (48, 96), classification head (512, GELU), Pre-Norm and Dropout (0.1) in `GlobalTransformerBlock`, heads (8, 16) in pooling and 16 in transformer, and 5-epoch LR warmup.

With these changes, the resulting accuracy on test dataset is now 94.98%, the highest achieved so far.
