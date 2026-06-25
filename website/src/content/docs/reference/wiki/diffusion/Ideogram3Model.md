---
title: "Ideogram3Model<T>"
description: "Ideogram 3 model for text-to-image generation with superior text rendering."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Ideogram 3 model for text-to-image generation with superior text rendering.

## For Beginners

Most image generators struggle to write readable text in images.
Ideogram 3 excels at this — it can generate signs, logos, posters, and business cards
with correctly spelled, properly formatted text.

How Ideogram 3 works:

1. Text prompt is parsed to identify embedded text requests (e.g., "a sign that says 'Hello'")
2. A text layout predictor determines optimal placement, font size, and orientation
3. The SiT diffusion backbone generates the image with text-aware conditioning
4. OCR-in-the-loop training ensures spelling accuracy

Key characteristics:

- Text layout prediction module for accurate text placement
- OCR-in-the-loop training for spelling accuracy
- SiT (Scalable Interpolant Transformer) backbone
- Handles multi-line text, different fonts, and curved text
- 16 latent channels

Advantages:

- Best-in-class text rendering accuracy
- Correct spelling even for complex words
- Natural text placement and sizing
- Handles various text styles (signs, posters, book covers)
- Good overall image quality beyond text

Limitations:

- API-only access
- Text rendering adds inference overhead
- Less control over exact font/style compared to dedicated design tools

## How It Works

Ideogram 3 specializes in generating images with accurate, legible text rendering.
It uses a specialized text layout prediction module alongside a SiT diffusion backbone
to ensure rendered text is spelled correctly, properly formatted, and placed naturally
within the generated scene.

Technical specifications:

- Architecture: SiT with text layout prediction
- Text layout module: Predicts bounding boxes, font size, orientation
- OCR-in-the-loop: Training-time OCR validation for spelling
- Backbone: SiT, hidden 2048, 24 layers
- VAE: 16 latent channels
- Default: 30 steps, guidance scale 7.5
- Resolution: 1024x1024 default

Reference: Ideogram Inc., "Ideogram 3", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Ideogram3Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,SiTPredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of Ideogram3Model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `SupportsTextRendering` | Gets whether this model has specialized text rendering capabilities. |
| `UsesOCRTraining` | Gets whether this model uses OCR-in-the-loop training for spelling accuracy. |
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
| `DefaultHeight` | Default image height for Ideogram 3 (1024x1024). |
| `DefaultWidth` | Default image width for Ideogram 3 (1024x1024). |

