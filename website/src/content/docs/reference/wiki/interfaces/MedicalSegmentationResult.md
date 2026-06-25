---
title: "MedicalSegmentationResult<T>"
description: "Result of medical image segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Result of medical image segmentation.

## Properties

| Property | Summary |
|:-----|:--------|
| `DiceScores` | Per-class Dice scores measuring segmentation quality (if ground truth was provided). |
| `Labels` | Per-pixel/voxel class labels. |
| `Probabilities` | Per-pixel/voxel class probabilities. |
| `Structures` | Metadata about each segmented structure (class name, volume, surface area, etc.). |

