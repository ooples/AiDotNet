---
title: "ReferenceOnlyModel<T>"
description: "Reference-Only model that uses a reference image's self-attention features for conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

Reference-Only model that uses a reference image's self-attention features for conditioning.

## For Beginners

Instead of using edge maps or depth maps as control,
this directly uses another image as a reference. The AI copies the style,
colors, and feel of the reference image into the new generation. No special
preprocessing of the reference image is needed.

## How It Works

Reference-Only control injects self-attention features from a reference image
into the denoising process without requiring a separate ControlNet encoder.
This enables style and content transfer by sharing attention keys and values
between the reference and generated images.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReferenceOnlyModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Double,Nullable<Int32>)` | Initializes a new Reference-Only model. |

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

