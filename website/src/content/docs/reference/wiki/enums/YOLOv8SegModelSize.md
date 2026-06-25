---
title: "YOLOv8SegModelSize"
description: "Defines the size variants for YOLOv8-Seg instance segmentation models."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the size variants for YOLOv8-Seg instance segmentation models.

## For Beginners

YOLOv8-Seg is a real-time instance segmentation model from Ultralytics.
Smaller sizes (N, S) are faster and suited for edge deployment, while larger sizes (L, X)
offer higher accuracy for server-side applications.

## How It Works

**Technical Details:** YOLOv8-Seg uses an anchor-free detection head with a YOLACT-style
prototype mask generation branch. The backbone is a CSPDarknet variant with C2f blocks.

**Reference:** Ultralytics, "YOLOv8", 2023.

## Fields

| Field | Summary |
|:-----|:--------|
| `L` | Large (46.0M params). |
| `M` | Medium (27.3M params). |
| `N` | Nano (3.4M params). |
| `S` | Small (11.8M params). |
| `X` | Extra-Large (71.8M params). |

