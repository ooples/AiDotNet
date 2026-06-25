---
title: "INoisePredictor<T>"
description: "Interface for noise prediction networks used in diffusion models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for noise prediction networks used in diffusion models.

## For Beginners

A noise predictor is like a "noise detective" that looks at a noisy image
and figures out exactly what noise was added to it.

How it works:

1. The model receives a noisy image and a timestep
2. The timestep tells the model how much noise should be in the image
3. The model predicts what noise pattern was added
4. This prediction is used to remove noise and recover the original image

Different architectures for noise prediction:

- U-Net: The original and most common, uses an encoder-decoder with skip connections
- DiT (Diffusion Transformer): Uses transformer blocks, powers state-of-the-art models like SD3 and Sora
- U-ViT: Hybrid of U-Net and Vision Transformer

The architecture choice affects:

- Quality of generated images
- Speed of generation
- Memory requirements
- Ability to scale to larger models

## How It Works

Noise predictors are the core neural networks in diffusion models that learn to predict
the noise added to samples at each timestep. They can be implemented as U-Nets,
Diffusion Transformers (DiT), or other architectures.

This interface extends `IFullModel` to provide all standard
model capabilities (training, saving, loading, gradients, checkpointing, etc.).

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseChannels` | Gets the base channel count used in the network architecture. |
| `ContextDimension` | Gets the expected context dimension for cross-attention conditioning. |
| `InputChannels` | Gets the number of input channels the predictor expects. |
| `OutputChannels` | Gets the number of output channels the predictor produces. |
| `SupportsCFG` | Gets whether this noise predictor supports classifier-free guidance. |
| `SupportsCrossAttention` | Gets whether this noise predictor supports cross-attention conditioning. |
| `TimeEmbeddingDim` | Gets the dimension of the time/timestep embedding. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTimestepEmbedding(Int32)` | Computes the timestep embedding for a given timestep. |
| `PredictNoise(Tensor<>,Int32,Tensor<>)` | Predicts the noise in a noisy sample at a given timestep. |
| `PredictNoiseWithEmbedding(Tensor<>,Tensor<>,Tensor<>)` | Predicts noise with explicit timestep embedding (for batched different timesteps). |

