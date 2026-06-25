---
title: "LiteVAEModel<T>"
description: "Lightweight VAE optimized for fast encoding/decoding on edge and mobile devices."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VAE`

Lightweight VAE optimized for fast encoding/decoding on edge and mobile devices.

## For Beginners

Standard VAEs are large and slow — fine for servers but too
slow for phones or real-time apps. LiteVAE uses efficient building blocks (like
depthwise-separable convolutions) to achieve nearly the same quality at a fraction
of the computational cost, enabling diffusion on mobile devices.

## How It Works

LiteVAE replaces heavy encoder/decoder blocks with depthwise-separable convolutions
and channel attention, achieving 3-5x faster encoding/decoding with minimal quality
loss. Designed for real-time applications and resource-constrained environments.

Reference: Sauer et al., "LiteVAE: Lightweight and Efficient Variational Autoencoders
for Latent Diffusion Models", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LiteVAEModel(Int32,Int32,Int32,ILossFunction<>,Nullable<Int32>)` | Initializes a new LiteVAE model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DownsampleFactor` |  |
| `InputChannels` |  |
| `LatentChannels` |  |
| `LatentScaleFactor` |  |
| `ParameterCount` |  |
| `SupportsTiling` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BackpropagateLossGradient(Tensor<>)` |  |
| `Clone` |  |
| `Decode(Tensor<>)` |  |
| `DeepCopy` |  |
| `Encode(Tensor<>,Boolean)` |  |
| `EncodeWithDistribution(Tensor<>)` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

