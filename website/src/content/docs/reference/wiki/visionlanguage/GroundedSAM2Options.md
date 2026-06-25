---
title: "GroundedSAM2Options"
description: "Configuration options for Grounded-SAM 2: combines Grounding DINO with SAM 2 for grounded segmentation and tracking."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Grounding`

Configuration options for Grounded-SAM 2: combines Grounding DINO with SAM 2 for grounded segmentation and tracking.

## For Beginners

These options configure the GroundedSAM2 model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GroundedSAM2Options(GroundedSAM2Options)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnableSegmentation` | Gets or sets whether to produce segmentation masks. |
| `EnableTracking` | Gets or sets whether to enable video object tracking. |

