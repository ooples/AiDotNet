---
title: "MatchLabel"
description: "Label indicating how an anchor should be treated during training."
section: "API Reference"
---

`Enums` · `AiDotNet.ComputerVision.Detection.Anchors`

Label indicating how an anchor should be treated during training.

## Fields

| Field | Summary |
|:-----|:--------|
| `Ignore` | Anchor is borderline - excluded from loss calculation. |
| `Negative` | Anchor not matched - should predict background/no object. |
| `Positive` | Anchor matched to a GT box - should predict the object. |

