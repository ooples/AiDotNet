---
title: "Llama32Vision<T>"
description: "Llama 3.2 Vision: Meta's 11B/90B vision models for edge/mobile deployment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Llama 3.2 Vision: Meta's 11B/90B vision models for edge/mobile deployment.

## For Beginners

Llama 3.2 Vision adds image understanding to Meta's popular
Llama 3.2 language model. Available in 11B and 90B parameter sizes, it uses a ViT vision
encoder connected through an MLP projection to the Llama decoder. The 11B variant is
particularly notable for being optimized for edge and mobile deployment — it can run on
devices like phones and laptops rather than requiring expensive cloud servers. The 90B
variant provides stronger performance for server-side use. Both models can describe images,
answer visual questions, and reason about visual content. Default values follow the original
paper settings.

## How It Works

Llama 3.2 Vision (Meta, 2024) extends the Llama 3.2 language model with vision capabilities,
available in 11B and 90B parameter variants. It uses a ViT vision encoder with MLP projection
to integrate visual features into the Llama decoder, optimized for edge and mobile deployment.

**References:**

- Paper: "The Llama 3 Herd of Models" (Meta, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Llama 3.2 Vision's cross-attention adapter architecture. |

