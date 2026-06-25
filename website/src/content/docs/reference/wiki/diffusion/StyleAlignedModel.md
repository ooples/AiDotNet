---
title: "StyleAlignedModel<T>"
description: "Style-Aligned model for consistent style across multiple generated images."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

Style-Aligned model for consistent style across multiple generated images.

## For Beginners

When you generate multiple images, this model makes them
all look like they belong to the same "art collection" — same colors, same style,
same artistic feel. It's like having one artist draw several different scenes.

## How It Works

Style-Aligned uses shared self-attention across multiple generated images during
the denoising process to ensure consistent style. This enables generating sets of
images that share the same artistic style without explicit style transfer.

Reference: Hertz et al., "Style Aligned Image Generation via Shared Attention", CVPR 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StyleAlignedModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Double,Nullable<Int32>)` | Initializes a new Style-Aligned model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

