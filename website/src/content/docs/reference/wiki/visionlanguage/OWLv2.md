---
title: "OWLv2<T>"
description: "OWLv2: self-training for scaling open-vocabulary detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Grounding`

OWLv2: self-training for scaling open-vocabulary detection.

## For Beginners

OWLv2 is an improved version of OWL-ViT that achieves better
open-vocabulary detection through self-training at web scale. Default values follow the
original paper settings.

## How It Works

OWLv2 (Minderer et al., 2023) scales open-vocabulary object detection through self-training
on web-scale image-text data. Building on OWL-ViT, it uses a stronger ViT-L/14 backbone
with 960px input resolution, pseudo-label self-training from an OWL-ViT teacher model, and
objectness-calibrated scoring for better separation of true objects from background patches.

**References:**

- Paper: "Scaling Open-Vocabulary Object Detection" (Google, 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GroundText(Tensor<>,String)` | Grounds text using OWLv2's scaled open-vocabulary detection approach. |

