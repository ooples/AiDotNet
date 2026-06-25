---
title: "KimiVL<T>"
description: "Kimi-VL: MoE VLM with MoonViT and long-context processing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Reasoning`

Kimi-VL: MoE VLM with MoonViT and long-context processing.

## For Beginners

Kimi-VL is a vision-language model with MoE architecture
for efficient multimodal reasoning. Default values follow the original paper settings.

## How It Works

Kimi-VL (Moonshot AI, 2025) is a Mixture of Experts vision-language model featuring
the MoonViT visual encoder and long-context processing capabilities. It uses sparse
MoE routing for efficient scaling and supports extended context windows for processing
high-resolution images and lengthy multi-turn conversations with visual inputs.

**References:**

- Paper: "Kimi-VL Technical Report" (Moonshot AI, 2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from image using Kimi-VL's MoE architecture with MoonViT encoder. |
| `ReasonWithChainOfThought(Tensor<>,String)` | Generates reasoning using Kimi-VL's MoE-based chain-of-thought with long-context. |

