---
title: "ControlNetTileModel<T>"
description: "ControlNet Tile model for upscaling and detail enhancement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

ControlNet Tile model for upscaling and detail enhancement.

## For Beginners

Give this model a blurry or low-resolution image, and it
will generate a sharp, detailed version while keeping the same colors and layout.
It's great for making images look clearer and more detailed.

## How It Works

ControlNet Tile preserves overall color and composition from a blurred/downsampled
input while regenerating fine details. Commonly used for image upscaling,
detail enhancement, and texture regeneration.

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

