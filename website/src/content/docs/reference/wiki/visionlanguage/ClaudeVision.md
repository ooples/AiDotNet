---
title: "ClaudeVision<T>"
description: "Claude Vision: reference implementation of Anthropic's multimodal reasoning model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Proprietary`

Claude Vision: reference implementation of Anthropic's multimodal reasoning model.

## For Beginners

Claude Vision is a proprietary multimodal model from Anthropic
with strong visual reasoning and document understanding. Default values follow the model's
recommended settings.

## How It Works

Claude Vision is a reference implementation of Anthropic's multimodal reasoning model.
The Claude 3/4 family features strong document and chart understanding, extended thinking
for complex visual reasoning, and processes images alongside text in a unified transformer
architecture with safety-focused RLHF training.

**References:**

- Claude 3/4 Vision: strong document and chart understanding with extended thinking (Anthropic, 2024-2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from an image using Claude's multimodal reasoning architecture. |

