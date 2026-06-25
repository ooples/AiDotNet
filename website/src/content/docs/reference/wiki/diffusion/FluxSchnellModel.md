---
title: "FluxSchnellModel<T>"
description: "FLUX.1 Schnell for ultra-fast 1-4 step generation from the FLUX architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

FLUX.1 Schnell for ultra-fast 1-4 step generation from the FLUX architecture.

## For Beginners

FLUX.1 is one of the best open-source image generators.
FLUX.1 Schnell (German for "fast") is its speed-optimized version that generates
images in just 1-4 steps. It's free for commercial use and produces remarkably good
images for a distilled model.

## How It Works

FLUX.1 Schnell is the distilled variant of FLUX.1 optimized for 1-4 step generation.
Uses the same double-stream transformer architecture as FLUX.1 Dev but distilled for
speed. Requires no classifier-free guidance, producing high-quality images instantly.

Reference: Black Forest Labs, "FLUX.1 Schnell", 2024

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

