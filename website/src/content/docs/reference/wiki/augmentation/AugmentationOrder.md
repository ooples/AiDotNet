---
title: "AugmentationOrder"
description: "Specifies how multiple augmentations are applied in a pipeline."
section: "API Reference"
---

`Enums` · `AiDotNet.Augmentation`

Specifies how multiple augmentations are applied in a pipeline.

## Fields

| Field | Summary |
|:-----|:--------|
| `OneOf` | Apply only one randomly selected augmentation. |
| `Random` | Randomly shuffle the order of augmentations each time. |
| `Sequential` | Apply augmentations in the order they were added. |

