---
title: "IDallE3Model<T>"
description: "Defines the contract for DALL-E 3-style text-to-image generation models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for DALL-E 3-style text-to-image generation models.

## For Beginners

DALL-E 3 creates images from text descriptions!

Key capabilities:

- High-fidelity image generation from text prompts
- Accurate text rendering within images
- Complex scene composition with multiple objects
- Style control (vivid vs natural)
- Multiple aspect ratios and sizes

Architecture concepts:

1. Text Encoder: Understands and expands prompts
2. Diffusion Model: Generates images through iterative denoising
3. Safety Systems: Filters inappropriate content
4. Quality Enhancement: Upscaling and refinement

## How It Works

DALL-E 3 represents a significant advancement in text-to-image generation,
with improved prompt following, text rendering, and overall image quality.
It uses a combination of diffusion models and language understanding.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxPromptLength` | Gets the maximum prompt length in characters. |
| `SupportedSizes` | Gets the supported image sizes. |
| `SupportsEditing` | Gets whether the model supports image editing (inpainting). |
| `SupportsVariations` | Gets whether the model supports image variations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CheckPromptSafety(String)` | Checks if a prompt is likely to be rejected for safety reasons. |
| `CreateVariations(Tensor<>,Int32,Double,DallE3ImageSize)` | Generates variations of an existing image. |
| `Edit(Tensor<>,Tensor<>,String,DallE3ImageSize)` | Edits an existing image based on a prompt and mask. |
| `EstimateQuality(String)` | Estimates the generation quality before actually generating. |
| `ExpandPrompt(String,DallE3Style)` | Expands a simple prompt into a more detailed description. |
| `Generate(String,DallE3ImageSize,DallE3Quality,DallE3Style,Nullable<Int32>)` | Generates an image from a text prompt. |
| `GenerateConsistentSet(String,IEnumerable<String>,Int32,DallE3ImageSize)` | Generates a consistent set of images (same character/scene, different poses/angles). |
| `GenerateForUseCase(String,String,DallE3ImageSize)` | Generates an image optimized for a specific use case. |
| `GenerateMultiple(String,Int32,DallE3ImageSize,DallE3Quality,DallE3Style)` | Generates multiple images from a text prompt. |
| `GenerateTileable(String,DallE3ImageSize)` | Generates a seamlessly tileable image. |
| `GenerateWithComposition(String,IEnumerable<ValueTuple<String,String,Double>>,DallE3ImageSize)` | Generates an image with controlled composition. |
| `GenerateWithPrompt(String,DallE3ImageSize,DallE3Quality,DallE3Style)` | Generates an image with the revised/expanded prompt returned. |
| `GenerateWithStyle(String,String,DallE3ImageSize,DallE3Quality)` | Generates an image in a specific artistic style. |
| `GenerateWithText(String,String,String,DallE3ImageSize)` | Generates an image with specific text rendered in it. |
| `Outpaint(Tensor<>,String,Int32,String)` | Outpaints an image, extending it beyond its original boundaries. |
| `Upscale(Tensor<>,Int32,Boolean)` | Upscales an image to higher resolution. |

