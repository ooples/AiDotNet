---
title: "Mask2FormerModelSize"
description: "Defines the backbone size variants for Mask2Former."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the backbone size variants for Mask2Former.

## For Beginners

Mask2Former is a universal segmentation model that handles semantic,
instance, and panoptic segmentation with a single architecture. The backbone size controls
feature extraction capacity. Swin-T is efficient for prototyping, while Swin-L offers
the highest accuracy.

## How It Works

**Technical Details:** Mask2Former uses a Swin Transformer or ResNet backbone with a
pixel decoder (Multi-Scale Deformable Attention) and a transformer decoder with masked
cross-attention for query-based mask prediction.

**Reference:** Cheng et al., "Masked-attention Mask Transformer for Universal Image
Segmentation", CVPR 2022.

## Fields

| Field | Summary |
|:-----|:--------|
| `R101` | ResNet-101 backbone (63M params). |
| `R50` | ResNet-50 backbone (44M params). |
| `SwinBase` | Swin-B backbone (107M params). |
| `SwinLarge` | Swin-L backbone (216M params). |
| `SwinSmall` | Swin-S backbone (69M params). |
| `SwinTiny` | Swin-T backbone (47M params). |

