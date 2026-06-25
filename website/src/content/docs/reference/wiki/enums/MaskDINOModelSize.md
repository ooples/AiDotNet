---
title: "MaskDINOModelSize"
description: "Defines the backbone size variants for Mask DINO."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the backbone size variants for Mask DINO.

## For Beginners

Mask DINO unifies object detection and segmentation by extending the
DINO detector with a mask prediction branch. It handles instance, panoptic, and semantic
segmentation in one architecture.

## How It Works

**Technical Details:** Built on DINO (DETR with Improved deNoising anchOr boxes) with
an additional mask branch. Uses ResNet or Swin Transformer backbones with deformable
attention in the transformer encoder-decoder.

**Reference:** Li et al., "Mask DINO: Towards A Unified Transformer-based Framework
for Object Detection and Segmentation", CVPR 2023.

## Fields

| Field | Summary |
|:-----|:--------|
| `R50` | ResNet-50 backbone (44M params). |
| `SwinLarge` | Swin-L backbone (218M params). |

