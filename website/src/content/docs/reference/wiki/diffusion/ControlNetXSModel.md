---
title: "ControlNetXSModel<T>"
description: "ControlNet-XS model — lightweight control network with minimal parameters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

ControlNet-XS model — lightweight control network with minimal parameters.

## For Beginners

ControlNet-XS is ControlNet but much smaller and faster:

Key differences from ControlNet:

- 50-100x fewer parameters than standard ControlNet
- Faster inference with minimal quality loss
- Simpler architecture: thin control encoder
- Works with SD 1.5 and SDXL

Use ControlNet-XS when you need:

- Spatial control with minimal compute overhead
- Edge/depth/pose-guided generation
- Mobile or edge deployment scenarios

## How It Works

ControlNet-XS is a significantly smaller variant of ControlNet that achieves comparable
control quality with only ~1% of the original ControlNet parameters. It uses a
streamlined encoder that directly injects control signals into the base model.

Technical specifications:

- Control encoder: lightweight copy with ~1% of base model parameters
- Compatible: SD 1.5, SD 2.x, SDXL
- Conditions: depth, canny edges, segmentation, pose
- Injection: direct feature addition (no zero convolutions)

Reference: Zavadski et al., "ControlNet-XS", 2024

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `ControlEncoder` | Gets the lightweight control encoder. |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

