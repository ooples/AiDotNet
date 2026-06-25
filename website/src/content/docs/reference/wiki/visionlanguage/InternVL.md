---
title: "InternVL<T>"
description: "InternVL: 6B InternViT with progressive alignment and LLaMA language backbone."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

InternVL: 6B InternViT with progressive alignment and LLaMA language backbone.

## For Beginners

InternVL uses a massive 6 billion parameter vision encoder
called InternViT-6B — one of the largest vision-only models ever built. It progressively
aligns this vision encoder with a LLaMA language model through a two-stage training
process: first contrastive learning (matching images to text) and then generative training
(learning to produce text from images). The huge vision encoder provides extremely rich
visual representations that capture both fine details and high-level semantics, making it
effective for a wide range of visual understanding tasks. Default values follow the
original paper settings.

## How It Works

InternVL (Chen et al., 2024) scales up the vision foundation model to 6B parameters (InternViT-6B)
and progressively aligns it with LLaMA through contrastive learning and generative training. The
large-scale vision encoder provides rich visual representations for instruction-following tasks.

**References:**

- Paper: "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks" (Chen et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using InternVL's progressive alignment architecture. |

