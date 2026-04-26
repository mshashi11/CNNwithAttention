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

### Mar 27, 2026 (Exp 41)

Changes incorporated:
1. Added `Dropout2d(0.05)` after each ReLU activation in the feature extraction layers for better regularization of convolutional features.
2. Maintained successful features from Exp 39: expanded channels to (64, 128), batch size 384, extra 3x3 `Conv2d` layer in the first block, learned `pos_embed` in `GlobalTransformerBlock`, classification head (512, GELU), Pre-Norm and Dropout (0.1) in `GlobalTransformerBlock`, and heads (8, 16).

With these changes, the resulting accuracy on test dataset is now 95.08%.

### Apr 14, 2026 (Exp 112)

After a series of experiments (Exp 44 to Exp 112), the model was significantly refined to achieve a new peak accuracy of **95.44%**.

Key improvements incorporated:
1. **Architectural Depth:** Increased depth in both stages. Block 1 now uses five 3x3 `Conv2d` layers, and Block 2 uses two 3x3 `Conv2d` layers.
2. **Channel Scaling:** Scaled Block 1 to 72 channels and Block 2 to 184 channels.
3. **Attention Optimization:** Optimized the individual attention head receptive fields.
   - Block 1: 6 heads (head_dim=12).
   - Block 2 and Global Reasoning: 4 heads (head_dim=46).
   - Larger head dimensions in later stages proved particularly effective for global reasoning.
4. **Classification Head:** Expanded the classification head to 1024 units with a Dropout of 0.20 and GELU activation.
5. **Regularization:** Fine-tuned `Dropout2d(0.02)` in features and added Gaussian Noise (0.02) to training inputs.
6. **Training Protocol:** Used a batch size of 256 and `CosineAnnealingLR` with `T_max=35`.

With these changes, the resulting accuracy on test dataset is now 95.44%, the highest achieved so far.
