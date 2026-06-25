---
title: "SelectionCriterion<T>"
description: "Defines how unlabeled samples are selected for labeling."
section: "API Reference"
---

`Enums` · `AiDotNet.Classification.SemiSupervised`

Defines how unlabeled samples are selected for labeling.

## Fields

| Field | Summary |
|:-----|:--------|
| `Threshold` | Select all samples above the confidence threshold. |
| `TopK` | Select the top-k most confident samples per iteration. |
| `TopKPerClass` | Select top-k samples per class to maintain class balance. |

