---
title: "ControlNetLiteModel<T>"
description: "Lightweight ControlNet model with reduced parameter count for faster inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

Lightweight ControlNet model with reduced parameter count for faster inference.

## For Beginners

This is a smaller, faster version of ControlNet. It uses
fewer parameters so it runs quicker and uses less memory, at the cost of slightly
less precise control signal adherence.

## How It Works

ControlNet Lite reduces the encoder to approximately 25% of the full ControlNet's
parameters while maintaining acceptable control quality. Suitable for real-time
or resource-constrained applications.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ControlNetLiteModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,ControlType,Nullable<Int32>)` | Initializes a new ControlNet Lite model. |

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

