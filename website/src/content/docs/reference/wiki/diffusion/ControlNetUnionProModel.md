---
title: "ControlNetUnionProModel<T>"
description: "ControlNet Union Pro model that supports multiple control types in a single model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

ControlNet Union Pro model that supports multiple control types in a single model.

## For Beginners

Instead of needing a different model file for edges, depth,
poses, etc., this single model handles all control types. You just tell it which
type of control image you're providing, and it adapts automatically.

## How It Works

ControlNet Union Pro consolidates multiple control types into a single unified model,
eliminating the need to load separate ControlNet checkpoints for each control type.
Supports switching between and combining control modes at inference time.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ControlNetUnionProModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,ControlType[],Nullable<Int32>)` | Initializes a new ControlNet Union Pro model. |

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

