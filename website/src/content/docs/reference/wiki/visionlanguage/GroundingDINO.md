---
title: "GroundingDINO<T>"
description: "Grounding DINO: open-set detection combining DINO with grounded pre-training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Grounding`

Grounding DINO: open-set detection combining DINO with grounded pre-training.

## For Beginners

Grounding DINO is a vision-language model that detects objects
in images based on text descriptions, enabling open-vocabulary detection without fixed
category lists. Default values follow the original paper settings.

## How It Works

Grounding DINO (Liu et al., 2024) combines the DINO detector with grounded pre-training
for open-set object detection. It uses a dual-encoder architecture with a Swin/ViT image
backbone and BERT-like text encoder, fused through cross-modal feature enhancement layers.
Learned object queries attend to both modalities via a cross-modal decoder, producing
text-conditioned bounding box detections with alignment scores.

**References:**

- Paper: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection" (IDEA, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GroundText(Tensor<>,String)` | Grounds a text query using Grounding DINO's cross-modal DETR architecture. |

