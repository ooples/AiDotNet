---
title: "InstaFlowModel<T>"
description: "InstaFlow model for one-step text-to-image via rectified flow straightening."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

InstaFlow model for one-step text-to-image via rectified flow straightening.

## For Beginners

Diffusion models follow curved paths from noise to images.
InstaFlow first "straightens" these paths (making them nearly straight lines), then
trains a model to jump from start to end in one step. It's like straightening a
winding road so you can drive directly to your destination instead of following curves.

## How It Works

InstaFlow straightens the probability flow ODE paths of a pretrained diffusion model
using the Reflow procedure, then distills the straightened model into a single-step
generator. The straighter paths make single-step generation possible with minimal
quality loss compared to multi-step generation.

Reference: Liu et al., "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation", ICLR 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InstaFlowModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new InstaFlow model. |

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

