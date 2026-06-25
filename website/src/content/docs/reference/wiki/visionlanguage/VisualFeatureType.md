---
title: "VisualFeatureType"
description: "Specifies the visual feature extraction method for foundational VLMs."
section: "API Reference"
---

`Enums` · `AiDotNet.VisionLanguage.Encoders`

Specifies the visual feature extraction method for foundational VLMs.

## Fields

| Field | Summary |
|:-----|:--------|
| `GridFeatures` | Grid features from a CNN backbone. |
| `PatchEmbeddings` | Raw image patches linearly embedded (e.g., ViLT, ViT-based). |
| `RegionFeatures` | Object-level features from a detection model (e.g., Faster R-CNN). |

