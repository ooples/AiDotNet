---
title: "DeepSeekVL2<T>"
description: "DeepSeek-VL2: MoE vision-language model with dynamic tiling and multi-head latent attention."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

DeepSeek-VL2: MoE vision-language model with dynamic tiling and multi-head latent attention.

## For Beginners

DeepSeek-VL2 is a major upgrade that combines Mixture-of-Experts
with dynamic image tiling. It has 64 total experts but only activates 6 per token, making
it efficient despite its large capacity. Dynamic tiling lets it handle images at variable
resolutions by splitting them into appropriately-sized tiles. It also uses multi-head latent
attention (MLA) to compress the key-value cache, reducing memory usage during inference.
Available in 1.8B, 4.5B, and 27B sizes for different compute budgets. Default values follow
the original paper settings.

## How It Works

DeepSeek-VL2 (Wu et al., 2024) advances multimodal understanding with Mixture-of-Experts (MoE)
architecture, dynamic image tiling for variable resolution input, and multi-head latent attention
for efficient KV cache compression. It uses 64 experts with 6 active per token.

**References:**

- Paper: "DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding" (Wu et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using DeepSeek-VL2's MoE + dynamic tiling architecture. |

