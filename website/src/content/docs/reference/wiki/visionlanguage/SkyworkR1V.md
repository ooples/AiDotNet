---
title: "SkyworkR1V<T>"
description: "Skywork R1V: cross-modal transfer of reasoning LLMs to vision."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Reasoning`

Skywork R1V: cross-modal transfer of reasoning LLMs to vision.

## For Beginners

Skywork R1V is a vision-language model that transfers reasoning
capabilities from text LLMs to multimodal understanding. Default values follow the original
paper settings.

## How It Works

Skywork R1V (2025) pioneers cross-modal transfer of reasoning capabilities from text-only
LLMs to vision-language models. It transfers chain-of-thought reasoning patterns learned
by text reasoning models to the visual domain, enabling structured multi-step reasoning
about images without requiring vision-specific reasoning data from scratch.

**References:**

- Paper: "Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought" (2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from image using Skywork R1V's cross-modal reasoning transfer. |
| `ReasonWithChainOfThought(Tensor<>,String)` | Generates multi-step reasoning using Skywork R1V's cross-modal transfer approach. |

