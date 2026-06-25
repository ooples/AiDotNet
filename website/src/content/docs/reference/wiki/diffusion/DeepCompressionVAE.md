---
title: "DeepCompressionVAE<T>"
description: "Deep Compression Autoencoder (DC-AE) for extremely high spatial compression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VAE`

Deep Compression Autoencoder (DC-AE) for extremely high spatial compression.

## For Beginners

Standard VAEs compress images 8x spatially (512x512 → 64x64).
DC-AE compresses 32x-128x (512x512 → 16x16 or even 4x4), making diffusion dramatically
faster because the model works on much smaller latent tensors. The two-stage training
ensures image quality doesn't suffer despite the extreme compression.

## How It Works

DC-AE achieves 32x-128x spatial compression (vs standard 8x) by combining residual
autoencoding with a decoupled two-stage training: first train a standard AE, then
add a lightweight latent adapter to achieve extreme compression while preserving quality.

Reference: Chen et al., "Deep Compression Autoencoder for Efficient High-Resolution
Diffusion Models", NeurIPS 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepCompressionVAE(Int32,Int32,Int32,Int32,ILossFunction<>,Nullable<Int32>)` | Initializes a new Deep Compression Autoencoder. |

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

