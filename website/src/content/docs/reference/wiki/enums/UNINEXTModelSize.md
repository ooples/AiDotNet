---
title: "UNINEXTModelSize"
description: "Defines the backbone size variants for UNINEXT."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the backbone size variants for UNINEXT.

## For Beginners

UNINEXT (Universal INstance pErception through neXt-generation learning)
reformulates 10+ instance perception tasks as a unified object discovery and retrieval problem.
It achieves state-of-the-art on over 20 benchmarks across object detection, instance
segmentation, referring expression comprehension, and more.

## How It Works

**Technical Details:** Uses a shared backbone with task-specific prompt embeddings.
Tasks include detection, instance segmentation, SOT, MOT, VIS, R-VOS, and more.
All are reformulated as retrieve-then-segment with unified query representations.

**Reference:** Yan et al., "Universal Instance Perception as Object Discovery and Retrieval",
CVPR 2023.

## Fields

| Field | Summary |
|:-----|:--------|
| `R50` | ResNet-50 backbone. |
| `SwinLarge` | Swin-L backbone. |
| `ViTHuge` | ViT-H backbone (Huge). |

