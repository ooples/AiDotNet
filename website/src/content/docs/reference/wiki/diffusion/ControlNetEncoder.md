---
title: "ControlNetEncoder<T>"
description: "ControlNet encoder per Zhang et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

ControlNet encoder per Zhang et al. (2023) "Adding Conditional Control to Text-to-Image Diffusion Models".
Uses convolutional layers (NOT dense/fully-connected) operating on spatial [C, H, W] tensors.
Zero convolutions are 1×1 convolutions initialized to zero for safe integration with pretrained models.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ControlNetEncoder(Int32,Int32,Int32[],Int32,Nullable<Int32>)` | Initializes a new ControlNetEncoder with convolutional layers per the paper. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the number of parameters in this encoder. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Encode(Tensor<>)` | Encodes a control image into multi-scale features. |
| `GetParameters` | Gets all parameters as a vector. |
| `SetParameters(Vector<>)` | Sets all parameters from a vector. |
| `ZeroInitializeConv(ConvolutionalLayer<>)` | Initializes a convolutional layer's weights and biases to zero (zero convolution per paper). |

