---
title: "NmsType"
description: "NMS (Non-Maximum Suppression) algorithm variants."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

NMS (Non-Maximum Suppression) algorithm variants.

## Fields

| Field | Summary |
|:-----|:--------|
| `DIoU` | DIoU-NMS - uses Distance-IoU for better localization. |
| `Soft` | Soft-NMS - reduces confidence of overlapping boxes instead of removing. |
| `Standard` | Standard hard NMS - removes overlapping boxes. |

