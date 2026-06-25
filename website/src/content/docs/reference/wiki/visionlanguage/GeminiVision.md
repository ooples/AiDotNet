---
title: "GeminiVision<T>"
description: "Gemini Vision: reference implementation of Google's native multimodal Mixture of Experts model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Proprietary`

Gemini Vision: reference implementation of Google's native multimodal Mixture of Experts model.

## For Beginners

Gemini Vision is a proprietary multimodal model from Google that
natively handles images, text, audio, and video. Default values follow the model's recommended
settings.

## How It Works

Gemini Vision is a reference implementation of Google's natively multimodal Mixture of Experts
model. The Gemini family processes interleaved text, image, audio, and video tokens natively
with 1M+ token context windows, using a sparse MoE architecture for efficient scaling across
multiple modalities.

**References:**

- Gemini: Google's natively multimodal model family with 1M+ token context (Google, 2024-2026)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from an image using Gemini's native multimodal MoE architecture. |

