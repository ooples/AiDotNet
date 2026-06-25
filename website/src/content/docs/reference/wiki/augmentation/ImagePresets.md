---
title: "ImagePresets<T>"
description: "Preset preprocessing configurations for common models."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Augmentation.Image`

Preset preprocessing configurations for common models.

## Methods

| Method | Summary |
|:-----|:--------|
| `CLIP` | CLIP model preprocessing: resize 224, center crop 224, normalize with CLIP stats. |
| `COCO(Int32)` | COCO detection preprocessing. |
| `DINO(Int32)` | DINO/DINOv2 preprocessing. |
| `ImageNet(Boolean)` | Standard ImageNet preprocessing: resize 256, center crop 224, normalize. |
| `SAM(Int32)` | SAM (Segment Anything Model) preprocessing. |
| `VOC(Int32)` | Pascal VOC preprocessing. |

