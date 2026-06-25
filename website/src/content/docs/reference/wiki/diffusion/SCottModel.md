---
title: "SCottModel<T>"
description: "SCott (Score Consistency via Optimal Transport) for efficient single/few-step generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

SCott (Score Consistency via Optimal Transport) for efficient single/few-step generation.

## For Beginners

SCott finds the most efficient path from noise to images using
optimal transport theory (the math of moving mass efficiently). Like finding the
shortest route for delivery trucks, SCott finds the shortest path from random noise
to your desired image, enabling generation in just 1-4 steps.

## How It Works

SCott combines score-based consistency training with optimal transport theory to learn
efficient single-step mappings. Uses a score-matching objective regularized by an
optimal transport penalty to produce straight transport paths that are well-suited
for few-step generation.

Reference: "SCott: Accelerating Diffusion Models with Stochastic Consistency Distillation", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SCottModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new SCott model. |

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

