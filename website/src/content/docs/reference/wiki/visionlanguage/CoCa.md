---
title: "CoCa<T>"
description: "CoCa (Contrastive Captioners): dual-loss model combining contrastive and captioning objectives."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Generative`

CoCa (Contrastive Captioners): dual-loss model combining contrastive and captioning objectives.

## For Beginners

CoCa combines two training objectives in one model: contrastive
learning (matching images to text descriptions) and captioning (generating text from images).
This dual approach produces strong image-text embeddings that work well for both retrieval
and generation tasks, making it a versatile foundation model. Default values follow the
original paper settings.

## How It Works

CoCa (Yu et al., TMLR 2022) combines contrastive learning with generative captioning in a single
model. The image encoder and the unimodal text decoder share a contrastive loss, while a multimodal
text decoder generates captions with cross-attention to image features.

**References:**

- Paper: "CoCa: Contrastive Captioners are Image-Text Foundation Models" (Yu et al., TMLR 2022)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using CoCa's dual contrastive-captioning architecture. |

