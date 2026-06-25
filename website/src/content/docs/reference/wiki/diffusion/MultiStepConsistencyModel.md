---
title: "MultiStepConsistencyModel<T>"
description: "Multi-Step Consistency Model that bridges single-step and multi-step generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

Multi-Step Consistency Model that bridges single-step and multi-step generation.

## For Beginners

Single-step generation is fast but can lack detail. This model
lets you trade speed for quality: use 1 step for real-time previews, 2-4 steps for
high-quality results. Each additional step refines the image, similar to how an artist
might do a rough sketch first, then add details in subsequent passes.

## How It Works

Extends consistency models to support configurable multi-step generation with improved
quality. While standard consistency models map noisy→clean in one step, this variant
chains multiple consistency steps with intermediate noise injection for better quality
at the cost of slightly more compute.

Reference: Based on Consistency Models (Song et al., 2023) with multi-step extensions

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiStepConsistencyModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new Multi-Step Consistency Model. |

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

