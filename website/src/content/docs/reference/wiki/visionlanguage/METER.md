---
title: "METER<T>"
description: "METER (Multimodal End-to-end TransformER) with systematic VLP component study."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Foundational`

METER (Multimodal End-to-end TransformER) with systematic VLP component study.

## For Beginners

METER is a vision-language model. Default values follow the original paper settings.

## How It Works

METER (Dou et al., CVPR 2022) is a systematic study of vision-language pre-training components.
It uses a CLIP ViT vision encoder and RoBERTa text encoder connected by co-attention transformer
fusion layers, providing an optimized combination of architecture choices for VLP.

**References:**

- Paper: "An Empirical Study of Training End-to-End Vision-and-Language Transformers" (Dou et al., CVPR 2022)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExtraTrainableLayers` |  |

