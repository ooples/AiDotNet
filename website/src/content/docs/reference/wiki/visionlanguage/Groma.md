---
title: "Groma<T>"
description: "Groma: localized visual tokenization for grounded understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Grounding`

Groma: localized visual tokenization for grounded understanding.

## For Beginners

Groma is a vision-language model that grounds language understanding
in specific image regions using localized visual tokens. Default values follow the original
paper settings.

## How It Works

Groma (Ma et al., 2024) introduces localized visual tokenization for grounding multimodal LLMs.
It uses a detect-then-describe pipeline with a region proposal network to extract candidate
object regions, RoI-Align for fixed-size region feature extraction, and quantized region tokens
interleaved with text tokens, enabling the LLM to refer to and describe specific image regions
without requiring explicit coordinate output in the text stream.

**References:**

- Paper: "Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models" (ByteDance, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GroundText(Tensor<>,String)` | Grounds text using Groma's localized visual tokenization approach. |

