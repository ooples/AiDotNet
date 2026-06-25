---
title: "GroundedSAM2<T>"
description: "Grounded-SAM 2: combines Grounding DINO with SAM 2 for grounded segmentation and tracking."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Grounding`

Grounded-SAM 2: combines Grounding DINO with SAM 2 for grounded segmentation and tracking.

## For Beginners

Grounded-SAM 2 is a vision-language model that combines text-based
object detection with high-quality segmentation and video tracking. Default values follow
the original paper settings.

## How It Works

Grounded-SAM 2 (Ren et al., 2024) combines Grounding DINO for text-conditioned object
detection with SAM 2 for high-quality mask generation and video object tracking. The two-stage
pipeline uses Grounding DINO's cross-modal DETR for bounding box detection, then SAM 2's
memory-augmented mask decoder for segmentation. SAM 2 also enables video object tracking
via memory attention across frames, supporting grounded tracking in videos.

**References:**

- Paper: "Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks" (IDEA, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GroundText(Tensor<>,String)` | Grounds text using a two-stage pipeline: Grounding DINO for detection + SAM2 for masks. |

