---
title: "LayerCategory"
description: "Classification of neural network layer types for automated per-layer decisions."
section: "API Reference"
---

`Enums` · `AiDotNet.Interfaces`

Classification of neural network layer types for automated per-layer decisions.

## How It Works

**For Beginners:** Neural networks are made of different types of layers, each with a
specific role. This enum categorizes layers so that tools like quantizers, pruners, and
pipeline schedulers can make smart per-layer decisions automatically.

For example, a quantizer might keep attention layers at 8-bit precision (they're sensitive
to quantization) while reducing dense layers to 4-bit (they're more tolerant).

## Fields

| Field | Summary |
|:-----|:--------|
| `Activation` | Activation function layers (ReLU, GELU, SiLU, Sigmoid, Tanh, etc.). |
| `Attention` | Self-attention, cross-attention, multi-head attention layers. |
| `Capsule` | Capsule network layers (PrimaryCapsule, DigitCapsule, routing). |
| `Convolution` | Convolutional layers (1D, 2D, 3D, depthwise, separable, dilated, deformable). |
| `Dense` | Dense/fully-connected layers (e.g., FullyConnectedLayer, DenseLayer). |
| `Embedding` | Embedding layers (token, positional, patch, time). |
| `FeedForward` | Feed-forward / MLP block layers. |
| `Gating` | Gating mechanisms (GLU, SwiGLU, GEGLU). |
| `Graph` | Graph neural network layers (GCN, GAT, GraphSAGE, GIN, message passing). |
| `Input` | Input layers. |
| `Memory` | Memory/external memory layers (NTM read/write heads). |
| `MixtureOfExperts` | Mixture of experts layers (MoE, Switch, Top-K routing). |
| `Normalization` | Normalization layers (BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RMSNorm). |
| `Other` | Custom, specialized, or unclassified layers. |
| `Pooling` | Pooling layers (MaxPool, AveragePool, AdaptivePool, GlobalPool). |
| `Positional` | Positional encoding layers (sinusoidal, ALiBi, RoPE, learned). |
| `Recurrent` | Recurrent layers (LSTM, GRU, RNN, bidirectional). |
| `Regularization` | Regularization layers (Dropout, GaussianNoise). |
| `Residual` | Residual/skip connection layers. |
| `StateSpaceModel` | State Space Model layers (Mamba, S4, S5, RWKV, RetNet). |
| `Structural` | Reshape, flatten, split, concatenate, and other structural layers. |
| `Transformer` | Transformer building blocks (EncoderLayer, DecoderLayer). |
| `Upsampling` | Deconvolution/upsampling layers (TransposedConv, PixelShuffle). |

