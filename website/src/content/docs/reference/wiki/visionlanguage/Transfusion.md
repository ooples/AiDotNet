---
title: "Transfusion<T>"
description: "Transfusion: combined autoregressive and diffusion loss in single transformer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Unified`

Transfusion: combined autoregressive and diffusion loss in single transformer.

## For Beginners

Transfusion is a unified model that combines autoregressive
text generation with diffusion-based image generation. Default values follow the original
paper settings.

## How It Works

Transfusion (Meta, 2024) combines autoregressive next-token prediction and continuous
diffusion loss within a single multi-modal transformer. Text tokens use cross-entropy loss
while image patches use diffusion denoising loss, allowing the model to both generate
text autoregressively and produce high-quality images through diffusion within a unified
architecture.

**References:**

- Paper: "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model" (Meta, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from image using Transfusion's mixed-modal transformer. |
| `GenerateImage(String)` | Generates an image from text using Transfusion's diffusion-within-transformer approach. |

