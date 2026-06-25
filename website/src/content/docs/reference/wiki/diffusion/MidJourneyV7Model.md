---
title: "MidJourneyV7Model<T>"
description: "MidJourney V7-style model architecture for artistic text-to-image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

MidJourney V7-style model architecture for artistic text-to-image generation.

## For Beginners

MidJourney V7 is renowned for producing the most visually stunning
and artistic images in the AI generation space.

How MidJourney V7 works:

1. Text is deeply interpreted by a proprietary language model for nuanced understanding
2. A multi-scale MMDiT-X processes tokens at different resolution scales
3. Aesthetic-aware training ensures consistently beautiful output
4. A stylize parameter controls artistic interpretation vs literal prompt following

Key characteristics:

- Multi-scale MMDiT-X for coherent generation across scales
- Proprietary language model for deep prompt interpretation
- Aesthetic-aware training with human preference data
- Stylize parameter (0-1000) controlling artistic expression
- Strong photorealistic and artistic capabilities
- 16 latent channels

Advantages:

- Best-in-class artistic and aesthetic quality
- Exceptional at photorealism and creative composition
- Strong prompt interpretation and creative expansion
- Stylize parameter gives fine control over output style

Limitations:

- API-only (not open-source, architecture details are proprietary)
- Less precise prompt adherence when stylize is high
- Requires internet connection and subscription

## How It Works

Represents the MidJourney V7-style architecture known for highly artistic and photorealistic
generation. Uses a multi-scale MMDiT-X architecture with aesthetic-aware training, enhanced
prompt interpretation via a proprietary language model, and a stylize parameter for controlling
the artistic vs photorealistic balance.

Technical specifications:

- Architecture: Multi-scale MMDiT-X with aesthetic training
- Hidden size: estimated ~4096, ~40+ layers
- Stylize range: 0 (literal) to 1000 (highly artistic)
- VAE: 16 latent channels
- Default: 50 steps, guidance scale 5.0
- Resolution: 1024x1024 default, aspect-ratio aware

Note: MidJourney is proprietary; this is a best-effort architectural representation
based on publicly available information.

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
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `SetParameters(Vector<>)` |  |

