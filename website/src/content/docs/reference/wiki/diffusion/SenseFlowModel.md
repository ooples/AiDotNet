---
title: "SenseFlowModel<T>"
description: "SenseFlow model for accelerated flow-matching generation via knowledge distillation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

SenseFlow model for accelerated flow-matching generation via knowledge distillation.

## For Beginners

Flow-matching models like FLUX generate amazing images but
need 20+ steps. SenseFlow is a distilled version that captures the same quality in
4-8 steps by learning to take "bigger strides" along the generation path, like a
student who learns shortcuts from a thorough teacher.

## How It Works

SenseFlow accelerates flow-matching models (like FLUX) through a combination of
progressive distillation and feature alignment. Maintains the straight-path ODE
formulation while reducing the number of steps needed from 20-50 to 4-8.

Reference: SenseTime, "SenseFlow: Accelerated Flow-Matching Generation", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SenseFlowModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,FluxDoubleStreamPredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new SenseFlow model. |

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

