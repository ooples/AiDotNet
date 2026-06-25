---
title: "ControlNetPlusPlusModel<T>"
description: "ControlNet++ model with improved conditioning via reward-guided training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

ControlNet++ model with improved conditioning via reward-guided training.

## For Beginners

ControlNet++ is a better version of ControlNet that follows
your control images (edges, depth, poses) more accurately. It was trained with
a smarter method that teaches it to match control signals more faithfully.

## How It Works

ControlNet++ improves upon ControlNet by using reward-guided training that
produces more consistent and higher-quality control signal adherence. It supports
multiple control types simultaneously with better composability.

Reference: Li et al., "ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback", ECCV 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ControlNetPlusPlusModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,ControlType,Double,Nullable<Int32>)` | Initializes a new ControlNet++ model. |

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

