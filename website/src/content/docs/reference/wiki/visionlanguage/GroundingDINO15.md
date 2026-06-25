---
title: "GroundingDINO15<T>"
description: "Grounding DINO 1.5: enhanced open-set detection with improved architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Grounding`

Grounding DINO 1.5: enhanced open-set detection with improved architecture.

## For Beginners

Grounding DINO 1.5 is an enhanced version of Grounding DINO with
a stronger backbone and multi-scale attention for improved open-set detection. Default values
follow the original paper settings.

## How It Works

Grounding DINO 1.5 (Ren et al., 2024) advances open-set object detection with a stronger
ViT-H backbone, multi-scale deformable attention for better handling of objects at different
scales, improved text-visual alignment with contrastive learning, and EfficientSAM integration
for segmentation-aware features. It builds on the original Grounding DINO architecture with
enhanced cross-modal fusion and larger-scale pre-training.

**References:**

- Paper: "Grounding DINO 1.5: Advance the Edge of Open-Set Object Detection" (IDEA, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GroundText(Tensor<>,String)` | Grounds a text query using Grounding DINO 1.5's enhanced architecture. |

