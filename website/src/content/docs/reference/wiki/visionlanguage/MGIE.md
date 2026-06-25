---
title: "MGIE<T>"
description: "MGIE: MLLM-guided image editing with LLaVA-based instruction understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Editing`

MGIE: MLLM-guided image editing with LLaVA-based instruction understanding.

## For Beginners

MGIE is a vision-language model that uses an LLM to understand
editing instructions and guide a diffusion model for precise image edits. Default values
follow the original paper settings.

## How It Works

MGIE (Apple, 2024) guides instruction-based image editing through a multimodal LLM (LLaVA)
that interprets and expands ambiguous editing instructions into detailed, expressive guidance.
The LLM-derived guidance conditions a diffusion model for semantically faithful edits that
align with user intent, bridging the gap between vague instructions and precise image modifications.

**References:**

- Paper: "Guiding Instruction-Based Image Editing via Multimodal Large Language Models" (Apple, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `EditImage(Tensor<>,String)` | Edits an image using MGIE's MLLM-guided expressive instruction pipeline. |

