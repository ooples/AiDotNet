---
title: "SAM21ModelSize"
description: "Defines the backbone size variants for SAM 2.1 (Segment Anything Model 2.1)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the backbone size variants for SAM 2.1 (Segment Anything Model 2.1).

## For Beginners

SAM 2.1 is an updated version of SAM 2 with refined training recipes
that improve segmentation accuracy. It uses the same Hiera backbone architecture as SAM 2
but with better-tuned checkpoints.

## How It Works

**Technical Details:** Same Hiera backbone as SAM 2 with improved training procedures.
Supports both image and video segmentation with memory attention for temporal consistency.

**Reference:** Ravi et al., "SAM 2: Segment Anything in Images and Videos", Meta AI, 2024.

## Fields

| Field | Summary |
|:-----|:--------|
| `BasePlus` | Base Plus variant (81M params). |
| `Large` | Large variant (224M params). |
| `Small` | Small variant (46M params). |
| `Tiny` | Tiny variant (39M params). |

