---
title: "Fuyu<T>"
description: "Fuyu: no vision encoder; raw patches directly into transformer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Fuyu: no vision encoder; raw patches directly into transformer.

## For Beginners

Fuyu takes the simplest possible approach to combining vision
and language: instead of using a separate vision encoder like CLIP or ViT, it feeds raw
image patches directly into the transformer decoder. Each image patch is simply projected
to the model's hidden dimension with a linear layer and treated like a text token. This
means the same transformer processes both image and text tokens, making the architecture
extremely simple. The trade-off is that it needs more compute to process images compared
to models with dedicated vision encoders, but it avoids any information loss from
pre-processing. Default values follow the original paper settings.

## How It Works

Fuyu (Adept, 2023) takes a radically simple approach to multimodal processing by feeding
raw image patches directly into the transformer decoder without a separate vision encoder.
Patches are linearly projected to the decoder dimension and interleaved with text tokens.

**References:**

- Paper: "Fuyu-8B: A Multimodal Architecture for AI Agents" (Adept, 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Fuyu's direct patch-to-transformer architecture. |

