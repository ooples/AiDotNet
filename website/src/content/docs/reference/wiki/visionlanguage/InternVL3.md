---
title: "InternVL3<T>"
description: "InternVL3: 78B SOTA open-source model with advanced training and test-time recipes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

InternVL3: 78B SOTA open-source model with advanced training and test-time recipes.

## For Beginners

InternVL3 is the largest and most capable model in the InternVL
series, with 78 billion total parameters. It uses InternViT-6B as the vision encoder and
InternLM3 as the language backbone with 8192 hidden dimensions and 80 decoder layers.
It achieves state-of-the-art performance among open-source models on challenging benchmarks
like MMMU (72.2 score). The model uses advanced training recipes and test-time strategies
(like chain-of-thought prompting) to push open-source multimodal models to new performance
levels. Default values follow the original paper settings.

## How It Works

InternVL3 (2025) is the largest open-source vision-language model with 78B parameters,
achieving 72.2 on MMMU (SOTA among open-source models). It uses InternViT-6B vision encoder
with InternLM3 language backbone at 8192 hidden dimension and 80 decoder layers.

**References:**

- Paper: "InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models" (2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using InternVL3's advanced training and test-time recipes. |

