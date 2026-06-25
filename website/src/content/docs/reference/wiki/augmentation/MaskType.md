---
title: "MaskType"
description: "Specifies the type of segmentation mask."
section: "API Reference"
---

`Enums` · `AiDotNet.Augmentation.Image`

Specifies the type of segmentation mask.

## Fields

| Field | Summary |
|:-----|:--------|
| `Binary` | Binary mask where each pixel is 0 or 1. |
| `Instance` | Instance segmentation where each object instance has a unique ID. |
| `Panoptic` | Panoptic segmentation combining semantic and instance. |
| `Semantic` | Semantic segmentation where each pixel has a class label. |

