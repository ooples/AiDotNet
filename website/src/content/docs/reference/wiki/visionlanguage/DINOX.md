---
title: "DINOX<T>"
description: "DINO-X: strongest open-world perception model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Grounding`

DINO-X: strongest open-world perception model.

## For Beginners

DINOX is a vision-language model that can detect and understand
objects using text descriptions, visual examples, or custom prompts. Default values follow
the original paper settings.

## How It Works

DINO-X (Ren et al., 2024) extends Grounding DINO with a universal object prompt module
that supports text, visual exemplars, and custom prompts. Trained on Grounding-100M
(100M+ images with long-tailed open-world categories), it uses a foundation encoder
with prompt-guided deformable cross-attention for open-world object detection and
understanding across diverse domains.

**References:**

- Paper: "DINO-X: A Unified Vision Model for Open-World Object Detection and Understanding" (IDEA, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GroundText(Tensor<>,String)` | Grounds a text query using DINO-X's universal perception architecture. |

