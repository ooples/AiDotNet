---
title: "LuminaT2XModel<T>"
description: "Lumina-T2X unified framework for transforming text into any modality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Lumina-T2X unified framework for transforming text into any modality.

## For Beginners

Lumina-T2X is a versatile model that can generate not just images,
but potentially video, audio, and 3D content too — all from text descriptions.

How Lumina-T2X works:

1. Text is encoded by Gemma 2B for broad language understanding
2. A shared Flag-DiT backbone processes the conditioning
3. Modality-specific heads produce output for the target format
4. Flow matching enables flexible resolution and duration handling

Key characteristics:

- Unified backbone for multi-modality generation
- Flag-DiT architecture with flow matching
- Gemma 2B text encoder
- Modality-specific output heads (image, video, audio, 3D)
- Resolution-agnostic and duration-agnostic design

Advantages:

- Single model backbone for multiple modalities
- Open-source and well-documented
- Flexible resolution and aspect ratio support
- Clean modular design for extension

## How It Works

Lumina-T2X is a unified framework for text-to-any generation using Flag-DiT (Flow-Aware
Generative DiT) blocks. It supports generating images, videos, 3D content, and audio from
text prompts using a shared transformer backbone with modality-specific output heads.
This implementation focuses on the image generation modality.

Reference: Gao et al., "Lumina-T2X: Transforming Text into Any Modality,
Resolution, and Duration via Flow-based Large Diffusion Transformers", 2024

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `SupportsMultiModality` | Gets whether this model supports multi-modality output. |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `SetParameters(Vector<>)` |  |

