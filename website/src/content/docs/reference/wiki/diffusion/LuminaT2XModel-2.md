---
title: "LuminaT2XModel<T>"
description: "Lumina-T2X model — transformer-based text-to-any generation (image, video, 3D, audio)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Lumina-T2X model — transformer-based text-to-any generation (image, video, 3D, audio).

## For Beginners

Lumina-T2X is a unified model for multiple generation tasks:

Key characteristics:

- Single architecture generates images, videos, 3D, and audio
- Flag-DiT: improved DiT with flow matching
- Resolution-aware encoding: handles any aspect ratio
- Gemma text encoder for multilingual prompts
- Scalable from 0.6B to 7B parameters

Use Lumina-T2X when you need:

- Multi-modal generation from text
- Flexible resolution/aspect ratio support
- Research into unified generation

## How It Works

Lumina-T2X is a unified transformer framework for generating multiple modalities
from text: images, videos, 3D objects, and audio. It uses a Flag-DiT backbone
with resolution-aware positional encoding.

Technical specifications:

- Architecture: Flag-DiT with flow matching
- Text encoder: Gemma-2B (2048-dim)
- Resolution: up to 2048x2048
- Latent channels: 4, 8x downsampling
- Flow matching with velocity prediction

Reference: Gao et al., "Lumina-T2X: Transforming Text into Any Modality", 2024

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
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

