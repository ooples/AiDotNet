---
title: "SD3FlashModel<T>"
description: "SD3 Flash for ultra-fast 1-4 step generation from SD3 architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

SD3 Flash for ultra-fast 1-4 step generation from SD3 architecture.

## For Beginners

SD3 Flash is to SD3 what FLUX Schnell is to FLUX — the same
great model but distilled to run in just 1-4 steps instead of 28. Perfect for
applications where speed matters more than squeezing out every bit of quality.

## How It Works

SD3 Flash is the consistency-distilled variant of Stable Diffusion 3 optimized for
1-4 step inference. Preserves SD3's strong text understanding and image quality
while dramatically reducing the number of required inference steps.

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

