---
title: "MiniGPT4<T>"
description: "MiniGPT-4: ViT + Q-Former aligned with Vicuna via single projection layer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

MiniGPT-4: ViT + Q-Former aligned with Vicuna via single projection layer.

## For Beginners

MiniGPT-4 was one of the first models to show that you can
connect a frozen image understanding system (ViT + Q-Former from BLIP-2) to a powerful
language model (Vicuna) with just a single linear projection layer and get impressive
results. Its two-stage training is simple: first learn on millions of image-text pairs
to align visual and text features, then fine-tune on a small set of curated instructions.
This minimal approach unlocked capabilities like detailed image descriptions, creative
writing from images, and visual reasoning that surprised the research community. Default
values follow the original paper settings.

## How It Works

MiniGPT-4 (Zhu et al., 2023) aligns a frozen ViT + Q-Former visual encoder (from BLIP-2)
with the Vicuna language model using a single linear projection layer. Two-stage training
(pretrain on image-text pairs, then fine-tune on curated instruction data) enables emergent
capabilities like detailed image description and creative writing from images.

**References:**

- Paper: "MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models" (Zhu et al., 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using MiniGPT-4's Q-Former + linear projection architecture. |
| `GetExtraTrainableLayers` |  |

