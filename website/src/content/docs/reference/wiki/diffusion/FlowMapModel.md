---
title: "FlowMapModel<T>"
description: "FlowMap model for one-step generation via flow-map learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

FlowMap model for one-step generation via flow-map learning.

## For Beginners

Standard flow models learn "how to move" from noise to images
and need multiple small steps. FlowMap learns "where to go" directly — given a noise
pattern, it outputs the final image location in one step. It's like learning the
destination directly instead of learning turn-by-turn directions.

## How It Works

FlowMap directly learns the transport map from noise to data distribution rather than
learning the velocity field and integrating it. This bypasses the need for ODE solving
entirely, enabling true one-step generation from any flow-matching model. Uses a
flow-map reparameterization that enables stable training.

Reference: "FlowMap: Learning to Generate High-Quality Samples with a Single Step", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlowMapModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new FlowMap model. |

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

