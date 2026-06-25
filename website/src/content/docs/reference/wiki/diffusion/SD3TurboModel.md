---
title: "SD3TurboModel<T>"
description: "SD3 Turbo for few-step generation from Stable Diffusion 3 via distillation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

SD3 Turbo for few-step generation from Stable Diffusion 3 via distillation.

## For Beginners

SD3 uses a powerful new architecture (MMDiT) that produces
excellent text rendering and composition. SD3 Turbo makes it 5-10x faster by
distilling it to just 4-8 steps while keeping the same high quality.

## How It Works

Distills the SD3 MMDiT architecture into a 4-8 step generator while preserving the
benefits of the triple text encoder (CLIP L, CLIP G, T5-XXL) conditioning. Uses
flow-matching-aware distillation adapted for the rectified flow formulation.

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

