---
title: "SegNeXtModelSize"
description: "Defines the model size variants for SegNeXt (Multi-Scale Convolutional Attention backbone)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the model size variants for SegNeXt (Multi-Scale Convolutional Attention backbone).

## For Beginners

SegNeXt comes in four sizes (Tiny through Large). Smaller sizes (Tiny)
are faster and use less memory, while larger sizes (Large) are more accurate but require
more compute. Tiny is great for real-time applications, while Base and Large offer
excellent accuracy for production deployments.

## How It Works

**Technical Details:** Each size uses a different MSCAN (Multi-Scale Convolutional Attention
Network) backbone with varying channel widths and encoder depths. All variants use the
Hamburger decoder for semantic segmentation.

**Reference:** Guo et al., "SegNeXt: Rethinking Convolutional Attention Design for
Semantic Segmentation", NeurIPS 2022.

## Fields

| Field | Summary |
|:-----|:--------|
| `Base` | MSCAN-B: Base variant (27.6M params). |
| `Large` | MSCAN-L: Large variant (48.9M params). |
| `Small` | MSCAN-S: Small variant (13.9M params). |
| `Tiny` | MSCAN-T: Tiny variant (4.3M params). |

