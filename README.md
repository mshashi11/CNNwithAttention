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
