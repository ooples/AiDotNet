---
title: "GIT<T>"
description: "GIT (Generative Image-to-Text): simple ViT encoder + autoregressive text decoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Generative`

GIT (Generative Image-to-Text): simple ViT encoder + autoregressive text decoder.

## For Beginners

GIT (Generative Image-to-Text) is a deliberately simple model
that connects a pre-trained image encoder (ViT) to a text decoder through just a linear
projection layer. Despite its simplicity — no Q-Former, perceiver, or cross-attention —
it achieves strong results on captioning and VQA by scaling data and model size. Default
values follow the original paper settings.

## How It Works

GIT (Wang et al., TMLR 2022) uses a simple architecture: a contrastive pre-trained image encoder
(ViT) connected to a text decoder via a linear projection. The decoder generates captions
autoregressively, conditioned on the projected visual features.

**References:**

- Paper: "GIT: A Generative Image-to-text Transformer for Vision and Language" (Wang et al., TMLR 2022)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using GIT's simple concatenation architecture. |

