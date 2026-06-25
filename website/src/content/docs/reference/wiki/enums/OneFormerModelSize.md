---
title: "OneFormerModelSize"
description: "Defines the backbone size variants for OneFormer."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the backbone size variants for OneFormer.

## For Beginners

OneFormer is a universal segmentation model that handles all three
segmentation tasks (semantic, instance, panoptic) with a single model trained only on
panoptic data. It uses text conditioning to switch between tasks at inference time.

## How It Works

**Technical Details:** Built on top of Mask2Former with a text encoder that conditions
the segmentation on a task description. Uses Swin or DiNAT (Dilated Neighborhood
Attention Transformer) backbones.

**Reference:** Jain et al., "OneFormer: One Transformer to Rule Universal Image
Segmentation", CVPR 2023.

## Fields

| Field | Summary |
|:-----|:--------|
| `DiNATLarge` | DiNAT-L backbone (223M params). |
| `SwinLarge` | Swin-L backbone (219M params). |

