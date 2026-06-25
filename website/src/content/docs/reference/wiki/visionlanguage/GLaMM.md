---
title: "GLaMM<T>"
description: "GLaMM: pixel-level grounded LMM generating text and segmentation masks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Grounding`

GLaMM: pixel-level grounded LMM generating text and segmentation masks.

## For Beginners

GLaMM is a vision-language model that can generate both text
descriptions and precise pixel-level segmentation masks for referred objects. Default values
follow the original paper settings.

## How It Works

GLaMM (Rasheed et al., 2024) is a pixel-level grounded large multimodal model that generates
natural language responses interleaved with segmentation masks. It uses a grounding image
encoder for multi-scale visual features, a pixel decoder for mask generation, and special
[SEG] tokens in the LLM output that trigger mask prediction through dot-product with
pixel-level feature embeddings, enabling fine-grained region-text alignment.

**References:**

- Paper: "GLaMM: Pixel Grounding Large Multimodal Model" (MBZUAI, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GroundText(Tensor<>,String)` | Grounds text using GLaMM's pixel-level grounding with mask generation. |

