---
title: "SDXLVAEModel<T>"
description: "SDXL-optimized VAE with improved decoder fidelity for 1024x1024 generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VAE`

SDXL-optimized VAE with improved decoder fidelity for 1024x1024 generation.

## For Beginners

SDXL generates 1024x1024 images (4x more pixels than SD 1.5's
512x512). The VAE decoder was retrained specifically for this higher resolution to
produce sharper, more detailed reconstructions. The encoder remains compatible with
the standard SD VAE, but the decoder is significantly improved.

## How It Works

The SDXL VAE is a fine-tuned version of the Stable Diffusion VAE with improved
decoder weights for higher fidelity reconstruction at 1024x1024 resolution. Uses
the same architecture as StandardVAE but with SDXL-specific scale factors and
optimized decoder weights trained on high-resolution data.

Reference: Podell et al., "SDXL: Improving Latent Diffusion Models for High-Resolution
Image Synthesis", ICLR 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SDXLVAEModel(Int32,Int32,Int32,Int32[],ILossFunction<>,Nullable<Int32>)` | Initializes a new SDXL VAE model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DownsampleFactor` |  |
| `InputChannels` |  |
| `LatentChannels` |  |
| `LatentScaleFactor` |  |
| `ParameterCount` |  |
| `SupportsSlicing` |  |
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

