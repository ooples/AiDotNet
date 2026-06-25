---
title: "ShowO2<T>"
description: "Show-o2: improved native unified multimodal model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Unified`

Show-o2: improved native unified multimodal model.

## For Beginners

Show-o2 is an improved version of Show-o with enhanced
understanding and generation quality. Default values follow the original paper settings.

## How It Works

Show-o2 (NUS, 2025) improves upon Show-o with enhanced native unified multimodal capabilities.
It refines the discrete diffusion approach for image generation with improved visual token
quality and strengthens the multimodal understanding pathway through better cross-modal
attention and scaled training on larger multimodal datasets.

**References:**

- Paper: "Show-o2: Improved Unified Multimodal Understanding and Generation" (NUS, 2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from image using Show-o2's improved unified transformer. |
| `GenerateImage(String)` | Generates an image from text using Show-o2's improved discrete diffusion. |

