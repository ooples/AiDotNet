---
title: "LLaVAOneVision<T>"
description: "LLaVA-OneVision: single model for images, multi-image, and videos with easy visual task transfer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

LLaVA-OneVision: single model for images, multi-image, and videos with easy visual task transfer.

## For Beginners

LLaVA-OneVision is a unified model that handles single images,
multiple images, and videos all in one architecture — previous models typically only
handled one type. It uses the SigLIP-SO400M vision encoder (a shape-optimized version
of SigLIP) with a Qwen2 language backbone. Through curriculum-based training — first
learning from single images, then multi-image tasks, then video — the model develops
strong visual understanding that transfers easily across different input types. This
means you can use the same model for photo analysis, comparing multiple images, and
understanding video content. Default values follow the original paper settings.

## How It Works

LLaVA-OneVision (Li et al., 2024) extends LLaVA-NeXT to handle single images, multi-image,
and video inputs in a single unified model. It uses SigLIP-SO400M vision encoder with
Qwen2 language backbone and achieves strong performance across image, multi-image, and video
benchmarks through curriculum-based visual instruction tuning.

**References:**

- Paper: "LLaVA-OneVision: Easy Visual Task Transfer" (Li et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using LLaVA-OneVision's unified image/multi-image/video architecture. |

