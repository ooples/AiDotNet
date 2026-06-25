---
title: "ShowO<T>"
description: "Show-o: single transformer for unified understanding and generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Unified`

Show-o: single transformer for unified understanding and generation.

## For Beginners

Show-o is a unified transformer model that handles both
multimodal understanding and image generation. Default values follow the original paper
settings.

## How It Works

Show-o (NUS, 2024) uses a single transformer to unify multimodal understanding and generation.
It combines autoregressive next-token prediction for text with discrete diffusion for image
generation within the same model, switching between generation modes based on the output
modality while sharing all transformer parameters.

**References:**

- Paper: "Show-o: One Single Transformer to Unify Multimodal Understanding and Generation" (NUS, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from image using Show-o's unified omni-attention transformer. |
| `GenerateImage(String)` | Generates an image from text using Show-o's discrete diffusion in token space. |

