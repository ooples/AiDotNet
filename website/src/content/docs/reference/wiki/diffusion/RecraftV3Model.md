---
title: "RecraftV3Model<T>"
description: "Recraft V3 model for professional-grade text-to-image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Recraft V3 model for professional-grade text-to-image generation.

## For Beginners

Recraft V3 is designed for professional use — creating marketing
materials, product images, and branded content.

How Recraft V3 works:

1. Text is encoded with style and layout conditioning
2. An MMDiT-X transformer generates the image with style-aware attention
3. Color palette conditioning ensures brand-consistent outputs
4. Text rendering module handles embedded text accurately

Key characteristics:

- Professional style presets (photo, illustration, vector, icon, etc.)
- Color palette control for brand consistency
- Superior text rendering in generated images
- MMDiT-X backbone with style-aware conditioning
- 16 latent channels

Advantages:

- Excellent for commercial and marketing content
- Strong text rendering capabilities
- Color palette consistency control
- Multiple style presets for different use cases
- Clean, professional output quality

Limitations:

- API-only access
- Style presets may limit creative freedom
- Optimized for commercial use cases, less for artistic expression

## How It Works

Recraft V3 focuses on professional and commercial-grade image generation with strong
control over style, composition, and brand consistency. It features style presets,
color palette control, and superior text rendering using an MMDiT-X backbone with
specialized conditioning for professional workflows.

Technical specifications:

- Architecture: MMDiT-X with style-aware conditioning
- Backbone: ~2B+ params, hidden 2048
- Text rendering: Specialized OCR-aligned text layout module
- Style presets: Photo, Digital Illustration, Vector, Icon, 3D Render
- Color control: RGB palette conditioning
- VAE: 16 latent channels, 8x spatial compression
- Default: 30 steps, guidance scale 7.0
- Resolution: 1024x1024 default, up to 2048x2048

Reference: Recraft.ai, "Recraft V3", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RecraftV3Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,MMDiTXNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of RecraftV3Model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `SupportsColorPalette` | Gets whether this model supports color palette conditioning. |
| `SupportsStylePresets` | Gets whether this model supports style presets. |
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

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultHeight` | Default image height for Recraft V3 (1024x1024). |
| `DefaultWidth` | Default image width for Recraft V3 (1024x1024). |

